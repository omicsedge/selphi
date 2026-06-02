//! Shared low-level BCF "typed atom" parsers.
//!
//! BCF encodes a typed value as a descriptor byte `(n_values << 4) | type_id`,
//! optionally followed by an overflow-length atom when `n_values == 15`, then the
//! payload. These helpers decode a single typed atom from an in-memory buffer at
//! offset `*o`, advancing `*o` past it.
//!
//! The same three readers were independently hand-copied in `srp::bcf_reader`,
//! `srp::bref3_writer` and `eval::accuracy`. Two of the three integer readers were
//! missing the multi-byte bounds checks → an out-of-bounds panic on a truncated
//! INFO/FORMAT atom. This is the single hardened copy that all three now share.
//!
//! Note the two string readers differ deliberately in how they strip NUL padding
//! (the SRP/BREF3 readers trim trailing NULs; the eval reader truncates at the
//! first NUL then does a lossy UTF-8 conversion) — both behaviors are preserved.

/// Read a BCF typed integer (type ids 1 = i8, 2 = i16, 3 = i32) from `buf` at
/// `*o`, advancing `*o` past it. Bounds-checked: a truncated atom yields 0 rather
/// than an out-of-bounds panic. Other type ids yield 0 (consuming only the
/// descriptor byte).
pub(crate) fn read_typed_i32(buf: &[u8], o: &mut usize) -> i32 {
    if *o >= buf.len() { return 0; }
    let tb = buf[*o]; *o += 1;
    match tb & 0x0F {
        1 => { if *o >= buf.len() { return 0; } let v = buf[*o] as i8 as i32; *o += 1; v }
        2 => { if *o + 2 > buf.len() { return 0; } let v = i16::from_le_bytes(buf[*o..*o+2].try_into().unwrap()) as i32; *o += 2; v }
        3 => { if *o + 4 > buf.len() { return 0; } let v = i32::from_le_bytes(buf[*o..*o+4].try_into().unwrap()); *o += 4; v }
        _ => 0
    }
}

/// Read a BCF typed string (type id 7) as a `String` with trailing NUL padding
/// trimmed; invalid UTF-8 yields "". A non-string atom advances `*o` past its
/// payload and returns "". Used by the SRP and BREF3 readers.
pub(crate) fn read_typed_str(buf: &[u8], o: &mut usize) -> String {
    if *o >= buf.len() { return String::new(); }
    let tb = buf[*o]; *o += 1;
    let tid = tb & 0x0F;
    let vl = { let r = (tb >> 4) as usize; if r == 15 { read_typed_i32(buf, o) as usize } else { r } };
    if tid == 7 {
        let e = (*o + vl).min(buf.len());
        let s = std::str::from_utf8(&buf[*o..e]).unwrap_or("").trim_end_matches('\0').to_string();
        *o = e; s
    } else {
        *o += vl * match tid { 1 => 1, 2 => 2, 3 => 4, 5 => 4, _ => 1 };
        String::new()
    }
}

/// Read a BCF typed string (type id 7) as raw bytes, truncated at the first NUL.
/// A non-string atom advances `*o` past its payload and returns empty. Used by the
/// eval reader (which then applies a lossy UTF-8 conversion).
pub(crate) fn read_typed_str_bytes(buf: &[u8], o: &mut usize) -> Vec<u8> {
    if *o >= buf.len() { return Vec::new(); }
    let tb = buf[*o]; *o += 1;
    let tid = tb & 0x0F;
    let vl = { let r = (tb >> 4) as usize; if r == 15 { read_typed_i32(buf, o) as usize } else { r } };
    if tid == 7 {
        let e = (*o + vl).min(buf.len());
        let s = &buf[*o..e];
        let end = s.iter().position(|&b| b == 0).unwrap_or(s.len());
        *o = e;
        s[..end].to_vec()
    } else {
        *o += vl * match tid { 1 => 1, 2 => 2, 3 => 4, 5 => 4, _ => 1 };
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_i32_widths() {
        // i8 = 5
        let mut o = 0; assert_eq!(read_typed_i32(&[0x11, 0x05], &mut o), 5); assert_eq!(o, 2);
        // i16 = 0x0102 = 258
        let mut o = 0; assert_eq!(read_typed_i32(&[0x12, 0x02, 0x01], &mut o), 258); assert_eq!(o, 3);
        // i32 = 0x00010001 = 65537
        let mut o = 0; assert_eq!(read_typed_i32(&[0x13, 0x01, 0x00, 0x01, 0x00], &mut o), 65537); assert_eq!(o, 5);
    }

    #[test]
    fn typed_i32_truncated_no_panic() {
        // i32 descriptor but only 2 payload bytes → 0 (was an OOB panic in the
        // unguarded copies).
        let mut o = 0; assert_eq!(read_typed_i32(&[0x13, 0x01, 0x00], &mut o), 0);
        // descriptor alone
        let mut o = 0; assert_eq!(read_typed_i32(&[0x12], &mut o), 0);
        // empty buffer
        let mut o = 0; assert_eq!(read_typed_i32(&[], &mut o), 0);
    }

    #[test]
    fn typed_str_variants() {
        // type 7, n=3, "ACG"
        let buf = [0x37u8, b'A', b'C', b'G'];
        let mut o = 0; assert_eq!(read_typed_str(&buf, &mut o), "ACG"); assert_eq!(o, 4);
        // trailing NUL padding is trimmed
        let buf = [0x47u8, b'A', b'C', 0, 0];
        let mut o = 0; assert_eq!(read_typed_str(&buf, &mut o), "AC"); assert_eq!(o, 5);
        // bytes form truncates at the first NUL
        let buf = [0x47u8, b'A', b'C', 0, b'X'];
        let mut o = 0; assert_eq!(read_typed_str_bytes(&buf, &mut o), b"AC".to_vec()); assert_eq!(o, 5);
    }

    #[test]
    fn typed_str_overflow_len() {
        // n=15 → read an i8 length atom (0x11,0x03) then 3 chars
        let buf = [0xF7u8, 0x11, 0x03, b'G', b'T', b'C'];
        let mut o = 0; assert_eq!(read_typed_str(&buf, &mut o), "GTC"); assert_eq!(o, 6);
    }
}
