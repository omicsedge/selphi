//! The merged VCF.gz and its `.tbi`, written through the shared tapped BGZF stack, under
//! temporary names that are renamed only once both are complete (a failure leaves no output that
//! looks complete), and the formatting of the lines the merge makes.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::srp::csi::{TappedBgzfWriter, TbiRec, TbiStreamIndexer};

use super::record::{Count, Rec, Src};

/// Uncompressed bytes between two index catch-ups.
const PROCESS_EVERY: usize = 4 << 20;

pub(super) struct OutWriter {
    w: Option<TappedBgzfWriter>,
    index: Option<TbiStreamIndexer>,
    path: PathBuf,
    tbi: PathBuf,
    tmp: PathBuf,
    tbi_tmp: PathBuf,
    prev: PathBuf,
    tbi_prev: PathBuf,
    /// TBI contig names (the contigs written, in order) and each header contig's index id.
    names: Vec<String>,
    ref_id: Vec<Option<u32>>,
    since: usize,
    pub lines: u64,
    /// Formatting buffer, reused for every line.
    pub buf: Vec<u8>,
    /// Both files are complete and named: nothing to clean up.
    published: bool,
}

fn with_suffix(p: &Path, s: &str) -> PathBuf { let mut o = p.as_os_str().to_owned(); o.push(s); PathBuf::from(o) }

/// `p` and every path its symbolic links lead through, in order (at most 40 links).
fn link_chain(p: &Path) -> Vec<PathBuf> {
    let mut chain = vec![p.to_path_buf()];
    while chain.len() <= 40 {
        let last = chain.last().expect("non-empty");
        let Ok(target) = std::fs::read_link(last) else { break };
        let next = if target.is_absolute() { target } else { last.parent().unwrap_or(Path::new(".")).join(target) };
        chain.push(next);
    }
    chain
}

/// Two paths naming the same directory entry (their directories resolved).
fn same_name(a: &Path, b: &Path) -> bool {
    let dir = |p: &Path| std::fs::canonicalize(p.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."))).ok();
    a.file_name() == b.file_name() && dir(a).is_some() && dir(a) == dir(b)
}

/// The temporary names `path` and its index are written under, and those a previous pair is kept
/// under while the new one takes its names.
pub fn temporary_names(path: &Path) -> [PathBuf; 4] {
    [with_suffix(path, ".tmp"), with_suffix(path, ".tbi.tmp"), with_suffix(path, ".prev"), with_suffix(path, ".tbi.prev")]
}

impl OutWriter {
    pub(super) fn create(path: &Path, workers: usize, n_contigs: usize) -> std::io::Result<Self> {
        let [tmp, tbi_tmp, prev, tbi_prev] = temporary_names(path);
        // Whatever holds these names now (a stale file, a link to another file) goes: creating a
        // file through a link would write the file it points to. A final name that IS one of those
        // files (a link to it) is refused first: removing it would remove the previous output.
        let tbi = with_suffix(path, ".tbi");
        let scratch = [&tmp, &tbi_tmp, &prev, &tbi_prev];
        for f in [path, tbi.as_path()] {
            if let Some(s) = link_chain(f).into_iter().skip(1).find(|p| scratch.iter().any(|s| same_name(s, p))) {
                return Err(crate::Fatal::error(2, format!("--calls: the output {} leads to {}, a name the merge writes its \
                    temporary files under", f.display(), s.display())).into());
            }
        }
        for p in scratch {
            if std::fs::symlink_metadata(p).is_ok() { std::fs::remove_file(p)?; }
        }
        let (w, index) = crate::srp::csi::tapped_bgzf_writer(&tmp, workers)?;
        Ok(OutWriter { w: Some(w), index: Some(index), path: path.to_path_buf(), tbi: with_suffix(path, ".tbi"), tmp, tbi_tmp, prev, tbi_prev,
            names: Vec::new(), ref_id: vec![None; n_contigs], since: 0, lines: 0, buf: Vec::with_capacity(4096), published: false })
    }

    #[inline]
    fn put(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.index.as_mut().expect("open").observe(bytes);
        self.w.as_mut().expect("open").write_all(bytes)?;
        self.since += bytes.len();
        if self.since >= PROCESS_EVERY {
            self.since = 0;
            self.index.as_mut().expect("open").process();
        }
        Ok(())
    }

    /// Header bytes (every line terminated).
    pub(super) fn header(&mut self, bytes: &[u8]) -> std::io::Result<()> { self.put(bytes) }

    /// One data line (`line` without its terminator; LF is added) of contig `contig` (its header
    /// index and name) at `pos`, spanning `rlen` bases for the index.
    pub(super) fn record(&mut self, contig: u32, name: &str, pos: i64, rlen: i64, line: &[u8]) -> std::io::Result<()> {
        let id = match self.ref_id[contig as usize] {
            Some(id) => id,
            None => {
                let id = self.names.len() as u32;
                self.names.push(name.to_string());
                self.ref_id[contig as usize] = Some(id);
                id
            }
        };
        self.index.as_mut().expect("open").push_meta(TbiRec { pos: pos - 1, rlen: rlen.clamp(1, u32::MAX as i64) as u32, ref_id: id });
        self.put(line)?;
        self.put(b"\n")?;
        self.lines += 1;
        Ok(())
    }

    /// The line formatted in [`Self::buf`].
    pub(super) fn record_buf(&mut self, contig: u32, name: &str, pos: i64, rlen: i64) -> std::io::Result<()> {
        let buf = std::mem::take(&mut self.buf);
        let r = self.record(contig, name, pos, rlen, &buf);
        self.buf = buf;
        r
    }

    /// Finish the file, write its index, and give both their names. On any failure no new file
    /// keeps its name, a previous pair is put back as it was, and the temporary files are removed.
    pub(super) fn finish(mut self) -> std::io::Result<()> {
        let mut w = self.w.take().expect("open");
        let mut file = w.finish()?.into_inner();
        file.flush()?;
        drop(file);
        self.index.take().expect("open").finish(&self.names, &self.tbi_tmp)?;
        self.publish()?;
        self.published = true;
        Ok(())
    }

    /// Move a previous pair aside, rename the new one into place, then drop the previous one. On a
    /// failure the new names are taken back and the previous pair restored; a failed restoration
    /// is reported with the error that caused it.
    fn publish(&self) -> std::io::Result<()> {
        use std::fs::{remove_file, rename};
        let pairs = [(&self.tmp, &self.path, &self.prev), (&self.tbi_tmp, &self.tbi, &self.tbi_prev)];
        let undo = |placed: usize, moved: &[bool; 2], e: std::io::Error| -> std::io::Error {
            let mut failed: Vec<String> = Vec::new();
            for (_, fin, _) in &pairs[..placed] { if let Err(x) = remove_file(fin) { failed.push(format!("{}: {x}", fin.display())); } }
            for (k, (_, fin, prev)) in pairs.iter().enumerate() {
                if moved[k] && let Err(x) = rename(prev, fin) { failed.push(format!("{} -> {}: {x}", prev.display(), fin.display())); }
            }
            if failed.is_empty() { e } else { std::io::Error::other(format!("{e}; restoring the previous output failed too: {}", failed.join("; "))) }
        };
        let mut moved = [false; 2];
        for (k, (_, fin, prev)) in pairs.iter().enumerate() {
            if std::fs::symlink_metadata(fin).is_ok() {
                if let Err(e) = rename(fin, prev) { return Err(undo(0, &moved, e)); }
                moved[k] = true;
            }
        }
        for (k, (tmp, fin, _)) in pairs.iter().enumerate() {
            if let Err(e) = rename(tmp, fin) { return Err(undo(k, &moved, e)); }
        }
        for (k, (_, _, prev)) in pairs.iter().enumerate() { if moved[k] { let _ = remove_file(prev); } }
        Ok(())
    }
}

impl Drop for OutWriter {
    /// A writer that did not publish (an error at any point) removes its temporary files.
    fn drop(&mut self) {
        if !self.published {
            drop(self.w.take());
            let _ = std::fs::remove_file(&self.tmp);
            let _ = std::fs::remove_file(&self.tbi_tmp);
        }
    }
}

#[inline]
fn push_int(buf: &mut Vec<u8>, v: i64) { let _ = write!(buf, "{v}"); }

#[inline]
fn push_count(buf: &mut Vec<u8>, c: Count) { match c { Some(v) => push_int(buf, v as i64), None => buf.push(b'.') } }

/// A Selphi row: the panel's identity, `GT:SRC:SDS:SGP:SRD:CST:CAC:CGQ:CPOS:CIDX`.
#[allow(clippy::too_many_arguments)]
pub(super) fn selphi_row(buf: &mut Vec<u8>, chrom: &str, pos: &str, id: &str, ref_a: &str, alt_a: &str,
    gt: &[u8], ds: f32, gp: [f32; 3], srd: u16, cst: &[u8], cac: Count, cgq: Option<i32>, members: &[(i64, u32)]) {
    use crate::lcwgs::output::push_fixed;
    buf.clear();
    for f in [chrom, pos, id, ref_a, alt_a] {
        buf.extend_from_slice(f.as_bytes());
        buf.push(b'\t');
    }
    buf.extend_from_slice(b".\tPASS\t.\tGT:SRC:SDS:SGP:SRD:CST:CAC:CGQ:CPOS:CIDX\t");
    buf.extend_from_slice(gt);
    buf.extend_from_slice(b":selphi:");
    push_fixed(buf, ds, 3);
    buf.push(b':');
    push_fixed(buf, gp[0], 3);
    buf.push(b',');
    push_fixed(buf, gp[1], 3);
    buf.push(b',');
    push_fixed(buf, gp[2], 3);
    buf.push(b':');
    push_int(buf, srd as i64);
    buf.push(b':');
    buf.extend_from_slice(cst);
    buf.push(b':');
    push_count(buf, cac);
    buf.push(b':');
    match cgq { Some(q) => push_int(buf, q as i64), None => buf.push(b'.') }
    buf.push(b':');
    if members.is_empty() {
        buf.extend_from_slice(b".:.");
    } else {
        for (n, &(p, _)) in members.iter().enumerate() { if n > 0 { buf.push(b','); } push_int(buf, p); }
        buf.push(b':');
        for (n, &(_, i)) in members.iter().enumerate() { if n > 0 { buf.push(b','); } push_int(buf, i as i64); }
    }
}

/// A compared and kept caller record: its content with `:SRC:SAC:SRD` appended to FORMAT and its
/// sample column, first padded with `.` to the FORMAT's length.
pub(super) fn annotated(buf: &mut Vec<u8>, rec: &Rec, src: Src) {
    buf.clear();
    let format = rec.field(8);
    if format == b"." || format.is_empty() {
        // No FORMAT fields: the merge's are the only ones.
        buf.extend_from_slice(&rec.line[..rec.f[8].s as usize]);
        buf.extend_from_slice(b"SRC:SAC:SRD\t");
    } else {
        buf.extend_from_slice(&rec.line[..rec.f[8].e as usize]);
        buf.extend_from_slice(b":SRC:SAC:SRD\t");
        let sample = rec.field(9);
        buf.extend_from_slice(sample);
        let n_format = format.iter().filter(|&&b| b == b':').count() + 1;
        let n_sample = sample.iter().filter(|&&b| b == b':').count() + 1;
        for _ in n_sample..n_format { buf.extend_from_slice(b":."); }
        buf.push(b':');
    }
    buf.extend_from_slice(src.name());
    buf.push(b':');
    for a in 0..rec.n_alt as usize {
        if a > 0 { buf.push(b','); }
        match rec.ann.get(a).copied().flatten() { Some((s, _)) => push_int(buf, s as i64), None => buf.push(b'.') }
    }
    buf.push(b':');
    for a in 0..rec.n_alt as usize {
        if a > 0 { buf.push(b','); }
        match rec.ann.get(a).copied().flatten() { Some((_, d)) => push_int(buf, d as i64), None => buf.push(b'.') }
    }
}

/// A fragment `[x, y]` of a block: its content with POS `x`, REF `ref_base`, INFO/END `y` (set or
/// added) and `SBLK=<POS>,<END>` of the input block appended to INFO.
pub(super) fn fragment(buf: &mut Vec<u8>, rec: &Rec, x: i64, y: i64, ref_base: u8) {
    buf.clear();
    buf.extend_from_slice(rec.field(0));
    buf.push(b'\t');
    push_int(buf, x);
    buf.push(b'\t');
    buf.extend_from_slice(rec.field(2));
    buf.push(b'\t');
    buf.push(ref_base);
    buf.push(b'\t');
    buf.extend_from_slice(&rec.line[rec.f[4].s as usize..rec.f[6].e as usize]);
    buf.push(b'\t');
    let info = rec.field(7);
    let mut first = true;
    let mut had_end = false;
    if info != b"." {
        for kv in info.split(|&b| b == b';') {
            if !first { buf.push(b';'); }
            first = false;
            if kv.starts_with(b"END=") {
                buf.extend_from_slice(b"END=");
                push_int(buf, y);
                had_end = true;
            } else {
                buf.extend_from_slice(kv);
            }
        }
    }
    if !had_end {
        if !first { buf.push(b';'); }
        buf.extend_from_slice(b"END=");
        push_int(buf, y);
    }
    buf.extend_from_slice(b";SBLK=");
    push_int(buf, rec.pos);
    buf.push(b',');
    push_int(buf, rec.stop);
    buf.push(b'\t');
    buf.extend_from_slice(&rec.line[rec.f[8].s as usize..]);
}
