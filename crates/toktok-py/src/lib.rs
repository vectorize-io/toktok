//! Python bindings for toktok. tiktoken-shaped API; data files ship inside the
//! wheel (`toktok/data/`) and are resolved by the package at import.

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PySet, PyString, PyTuple};
use std::collections::BTreeSet;

use toktok::Tokenizer as Core;

#[pyclass(name = "Tokenizer", module = "toktok._toktok", frozen)]
struct PyTokenizer {
    inner: Core,
}

/// Collect a Python `allowed_special` / `disallowed_special` argument into a set
/// of strings. `"all"` expands to every special the encoding knows.
fn to_set(
    obj: Option<&Bound<'_, PyAny>>,
    all: &[(String, u32)],
    default_all: bool,
) -> PyResult<BTreeSet<String>> {
    let Some(obj) = obj else {
        return Ok(if default_all {
            all.iter().map(|(s, _)| s.clone()).collect()
        } else {
            BTreeSet::new()
        });
    };
    if obj.is_none() {
        return Ok(BTreeSet::new());
    }
    if obj.is_instance_of::<PyString>() {
        if obj.extract::<String>()? == "all" {
            return Ok(all.iter().map(|(s, _)| s.clone()).collect());
        }
        return Err(PyValueError::new_err(
            "allowed_special/disallowed_special must be 'all' or a collection of strings",
        ));
    }
    let mut out = BTreeSet::new();
    for item in obj.try_iter()? {
        out.insert(item?.extract::<String>()?);
    }
    Ok(out)
}

fn u32s_to_bytes(py: Python<'_>, ids: &[u32]) -> Py<PyBytes> {
    let raw: &[u8] = unsafe {
        std::slice::from_raw_parts(ids.as_ptr() as *const u8, std::mem::size_of_val(ids))
    };
    PyBytes::new(py, raw).unbind()
}

#[pymethods]
impl PyTokenizer {
    #[new]
    #[pyo3(signature = (encoding, datadir = ""))]
    fn new(encoding: &str, datadir: &str) -> PyResult<Self> {
        // The bundled encodings are compiled into the extension module, so the
        // wheel ships no data files; a datadir is only for encodings you supply.
        let loaded = if datadir.is_empty() {
            Core::builtin(encoding)
        } else {
            Core::load_dir(datadir, encoding)
        };
        loaded
            .map(|inner| PyTokenizer { inner })
            .map_err(|e| PyRuntimeError::new_err(e.0))
    }

    /// tiktoken's `Encoding.encode`: raise if a disallowed special string occurs;
    /// otherwise encode, turning allowed specials into their ids.
    #[pyo3(signature = (text, allowed_special = None, disallowed_special = None))]
    fn encode(
        &self,
        py: Python<'_>,
        text: &str,
        allowed_special: Option<&Bound<'_, PyAny>>,
        disallowed_special: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<u32>> {
        let sp = self.inner.special_tokens();
        let allow = to_set(allowed_special, sp, false)?;
        let mut disallow = to_set(disallowed_special, sp, true)?;
        for a in &allow {
            disallow.remove(a);
        }
        for s in &disallow {
            if !s.is_empty() && text.contains(s.as_str()) {
                let q = format!("{s:?}");
                return Err(PyValueError::new_err(format!(
                    "Encountered text corresponding to disallowed special token {q}.\n\
                     If you want this text to be encoded as a special token, pass it to \
                     `allowed_special`, e.g. `allowed_special={{{q}, ...}}`.\n\
                     If you want this text to be encoded as normal text, disable the check \
                     for this token by passing `disallowed_special=()`, or use `encode_ordinary`."
                )));
            }
        }
        Ok(py.detach(|| {
            if allow.is_empty() {
                self.inner.encode(text.as_bytes())
            } else {
                self.inner.encode_allowed(text, |s| allow.contains(s))
            }
        }))
    }

    /// Encode ignoring special tokens entirely (tiktoken's `encode_ordinary`).
    fn encode_ordinary(&self, py: Python<'_>, text: &str) -> Vec<u32> {
        py.detach(|| self.inner.encode(text.as_bytes()))
    }

    /// Encode raw bytes (no UTF-8 validation).
    fn encode_bytes(&self, py: Python<'_>, data: &[u8]) -> Vec<u32> {
        py.detach(|| self.inner.encode(data))
    }

    /// Every special-token string is turned into its id.
    fn encode_with_special(&self, py: Python<'_>, text: &str) -> Vec<u32> {
        py.detach(|| self.inner.encode_with_special(text))
    }

    /// Same as `encode`, but returns the ids as a little-endian uint32 buffer —
    /// the Python wrapper turns it into a numpy array with no per-token Python
    /// int objects (the cost that otherwise dominates big inputs).
    #[pyo3(signature = (text, allowed_special = None, disallowed_special = None))]
    fn encode_to_buffer(
        &self,
        py: Python<'_>,
        text: &str,
        allowed_special: Option<&Bound<'_, PyAny>>,
        disallowed_special: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyBytes>> {
        let ids = self.encode(py, text, allowed_special, disallowed_special)?;
        Ok(u32s_to_bytes(py, &ids))
    }

    /// Same as `encode`, but returns a numpy `uint32` array — the fastest path
    /// from Python: no per-token Python int objects, just one buffer.
    #[pyo3(signature = (text, allowed_special = None, disallowed_special = None))]
    fn encode_to_numpy<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        allowed_special: Option<&Bound<'py, PyAny>>,
        disallowed_special: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let buf = self.encode_to_buffer(py, text, allowed_special, disallowed_special)?;
        frombuffer(py, buf.bind(py), "uint32")
    }

    /// Encode many texts in parallel into numpy arrays: `(ids, offsets)`, where
    /// text `i` occupies `ids[offsets[i]:offsets[i + 1]]`.
    #[pyo3(signature = (texts, threads = 0, with_special = false))]
    fn encode_batch_to_numpy<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
        threads: usize,
        with_special: bool,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let (flat, offsets) = self.encode_batch(py, texts, threads, with_special);
        Ok((
            frombuffer(py, flat.bind(py), "uint32")?,
            frombuffer(py, offsets.bind(py), "int64")?,
        ))
    }

    /// Ordinary ids + per-token spans. `unit="byte"` gives exact UTF-8 byte
    /// offsets (they tile the input); `unit="char"` gives code-point offsets
    /// (HF `offset_mapping` shape).
    #[pyo3(signature = (text, unit = "byte"))]
    fn encode_with_offsets(
        &self,
        py: Python<'_>,
        text: &str,
        unit: &str,
    ) -> PyResult<(Vec<u32>, Vec<(u32, u32)>)> {
        if unit != "byte" && unit != "char" {
            return Err(PyValueError::new_err("unit must be 'byte' or 'char'"));
        }
        Ok(py.detach(|| {
            let (ids, bounds) = self.inner.encode_with_offsets(text.as_bytes());
            let spans = if unit == "byte" {
                bounds.windows(2).map(|w| (w[0], w[1])).collect()
            } else {
                // code-point index of each byte position (count of UTF-8 lead bytes)
                let b = text.as_bytes();
                let mut b2c = vec![0u32; b.len() + 1];
                let mut c = 0u32;
                for i in 0..b.len() {
                    b2c[i] = c;
                    if b[i] & 0xC0 != 0x80 {
                        c += 1;
                    }
                }
                b2c[b.len()] = c;
                bounds
                    .windows(2)
                    .map(|w| {
                        let f = |x: u32| *b2c.get(x as usize).unwrap_or(&c);
                        (f(w[0]), f(w[1]))
                    })
                    .collect()
            };
            (ids, spans)
        }))
    }

    /// Exact bytes/str of a SINGLE token -> its id.
    fn encode_single_token(&self, piece: &Bound<'_, PyAny>) -> PyResult<u32> {
        let owned: Vec<u8> = if let Ok(b) = piece.cast::<PyBytes>() {
            b.as_bytes().to_vec()
        } else {
            piece.extract::<String>()?.into_bytes()
        };
        match self.inner.token_id(&owned) {
            id if id >= 0 => Ok(id as u32),
            _ => Err(PyKeyError::new_err(
                "bytes are not a single token in this encoding",
            )),
        }
    }

    fn count(&self, py: Python<'_>, text: &str) -> usize {
        py.detach(|| self.inner.count(text.as_bytes()))
    }

    /// Encode many texts in parallel. Returns (flat uint32 id buffer, per-text
    /// int64 offsets buffer with `len(texts)+1` entries).
    #[pyo3(signature = (texts, threads = 0, with_special = false))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        threads: usize,
        with_special: bool,
    ) -> (Py<PyBytes>, Py<PyBytes>) {
        let (flat, offsets) = py.detach(|| {
            let refs: Vec<&[u8]> = texts.iter().map(|s| s.as_bytes()).collect();
            let per = self.inner.encode_batch(&refs, threads, with_special);
            let total: usize = per.iter().map(|v| v.len()).sum();
            let mut flat = Vec::with_capacity(total);
            let mut offsets = Vec::with_capacity(per.len() + 1);
            offsets.push(0i64);
            for v in &per {
                flat.extend_from_slice(v);
                offsets.push(flat.len() as i64);
            }
            (flat, offsets)
        });
        let off_raw: &[u8] = unsafe {
            std::slice::from_raw_parts(
                offsets.as_ptr() as *const u8,
                std::mem::size_of_val(&offsets[..]),
            )
        };
        (u32s_to_bytes(py, &flat), PyBytes::new(py, off_raw).unbind())
    }

    /// Token counts for many texts, in parallel — no ids are materialized, so a
    /// counting workload allocates O(threads) instead of O(total tokens).
    #[pyo3(signature = (texts, threads = 0, with_special = false))]
    fn count_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        threads: usize,
        with_special: bool,
    ) -> Vec<u32> {
        py.detach(|| {
            let refs: Vec<&[u8]> = texts.iter().map(|s| s.as_bytes()).collect();
            self.inner.count_batch(&refs, threads, with_special)
        })
    }

    #[pyo3(signature = (ids, errors = "replace"))]
    fn decode(&self, py: Python<'_>, ids: Vec<u32>, errors: &str) -> PyResult<String> {
        let bytes = py.detach(|| self.inner.decode(&ids));
        match errors {
            "strict" => String::from_utf8(bytes)
                .map_err(|e| PyValueError::new_err(format!("invalid utf-8 in decode: {e}"))),
            "ignore" => Ok(String::from_utf8_lossy(&bytes).replace('\u{FFFD}', "")),
            _ => Ok(String::from_utf8_lossy(&bytes).into_owned()),
        }
    }

    fn decode_bytes(&self, py: Python<'_>, ids: Vec<u32>) -> Py<PyBytes> {
        let bytes = py.detach(|| self.inner.decode(&ids));
        PyBytes::new(py, &bytes).unbind()
    }

    fn decode_single_token_bytes(&self, py: Python<'_>, id: u32) -> PyResult<Py<PyBytes>> {
        if let Some(b) = self.inner.token_bytes(id) {
            return Ok(PyBytes::new(py, b).unbind());
        }
        for (s, sid) in self.inner.special_tokens() {
            if *sid == id {
                return Ok(PyBytes::new(py, s.as_bytes()).unbind());
            }
        }
        Err(PyKeyError::new_err(format!("unknown token id {id}")))
    }

    #[pyo3(signature = (batch, errors = "replace"))]
    fn decode_batch(
        &self,
        py: Python<'_>,
        batch: Vec<Vec<u32>>,
        errors: &str,
    ) -> PyResult<Vec<String>> {
        batch
            .into_iter()
            .map(|ids| self.decode(py, ids, errors))
            .collect()
    }

    /// All base-vocab token byte strings, indexed by id.
    fn token_byte_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let items = (0..self.inner.vocab_size() as u32)
            .map(|id| PyBytes::new(py, self.inner.token_bytes(id).unwrap()));
        PyList::new(py, items).unwrap()
    }

    fn is_special_token(&self, id: u32) -> bool {
        self.inner.special_tokens().iter().any(|(_, s)| *s == id)
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.encoding()
    }

    /// Exact heap footprint of the loaded encoding, in bytes.
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    #[getter]
    fn n_vocab(&self) -> usize {
        self.inner.n_vocab()
    }

    #[getter]
    fn max_token_value(&self) -> usize {
        self.inner.n_vocab() - 1
    }

    #[getter]
    fn eot_token(&self) -> PyResult<u32> {
        for (s, id) in self.inner.special_tokens() {
            if s == "<|endoftext|>" {
                return Ok(*id);
            }
        }
        Err(PyKeyError::new_err("this encoding has no <|endoftext|>"))
    }

    #[getter]
    fn special_tokens_set<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        PySet::new(py, self.inner.special_tokens().iter().map(|(s, _)| s.as_str()))
    }

    /// `{special string: id}` — tiktoken's `_special_tokens`.
    fn special_tokens<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.inner.special_tokens().iter().map(|(s, id)| {
                PyTuple::new(py, [s.into_pyobject(py).unwrap().into_any(), id.into_pyobject(py).unwrap().into_any()]).unwrap()
            }),
        )
    }

    fn __repr__(&self) -> String {
        format!("<toktok.Tokenizer '{}'>", self.inner.encoding())
    }
}

/// numpy.frombuffer over a bytes object — a read-only zero-copy view.
fn frombuffer<'py>(
    py: Python<'py>,
    buf: &Bound<'py, PyBytes>,
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy").map_err(|_| {
        PyRuntimeError::new_err(
            "the numpy methods need numpy installed: pip install 'toktok-rs[numpy]'",
        )
    })?;
    np.call_method1("frombuffer", (buf, dtype))
}

#[pymodule]
fn _toktok(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("BUILTIN_ENCODINGS", toktok::BUILTIN_ENCODINGS.to_vec())?;
    Ok(())
}
