//! Cross-chromosome reference profile save/load.
//!
//! Format: binary, per-sample top-50 ref individuals (as haplotype indices).

/// Save per-sample reference haplotype usage profile after phasing.
pub fn save_ref_profile(path: &str, profiles: &[Vec<(u32, usize)>], n_samples: usize) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let top_k: u32 = 50;
    f.write_all(&(n_samples as u32).to_le_bytes())?;
    f.write_all(&top_k.to_le_bytes())?;
    for si in 0..n_samples {
        let prof = &profiles[si];
        for i in 0..top_k as usize {
            let (count, ref_ind) = if i < prof.len() { prof[i] } else { (0, 0) };
            f.write_all(&(ref_ind as u32).to_le_bytes())?;
            f.write_all(&count.to_le_bytes())?;
        }
    }
    f.flush()
}

/// Load reference haplotype profile to seed conditioning set.
pub fn load_ref_profile(path: &str, n_samples: usize) -> std::io::Result<Vec<Vec<usize>>> {
    use std::io::Read;
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)?; let saved_n = u32::from_le_bytes(buf4) as usize;
    f.read_exact(&mut buf4)?; let top_k = u32::from_le_bytes(buf4) as usize;
    if saved_n != n_samples {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("Profile has {} samples but input has {}", saved_n, n_samples)));
    }
    let mut result = Vec::with_capacity(n_samples);
    for _si in 0..n_samples {
        let mut preferred = Vec::new();
        for _i in 0..top_k {
            f.read_exact(&mut buf4)?; let ref_ind = u32::from_le_bytes(buf4) as usize;
            f.read_exact(&mut buf4)?; let count = u32::from_le_bytes(buf4);
            if count > 0 {
                // Convert ref individual index to haplotype indices (both haps)
                let h0 = n_samples * 2 + ref_ind * 2;
                let h1 = h0 + 1;
                preferred.push(h0);
                preferred.push(h1);
            }
        }
        result.push(preferred);
    }
    Ok(result)
}
