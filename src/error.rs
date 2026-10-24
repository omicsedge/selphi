//! The error that ends a run.
//!
//! Library code never exits the process. Where a run cannot go on (an unreadable
//! input, a refused configuration), the library reports the message exactly as the
//! command line prints it and returns a [`Fatal`]: the message and the exit status
//! the `selphi` binary exits with ([`Fatal::code`]); a program that embeds the
//! library gets the value and decides.
//!
//! A [`Fatal`] has always been reported where it was raised, so whoever receives
//! one must not print it again. Functions whose error type is `std::io::Error`
//! carry it inside one (`From<Fatal> for io::Error`), and [`Fatal::of`] finds it.

use std::fmt;
use std::io;

/// A run that cannot go on. See the module documentation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fatal {
    code: i32,
    message: String,
}

impl Fatal {
    /// Report `message` as an error (`selphi_error!`: `ERROR: <message>` on stderr
    /// and in the run's `.log` file) and return it with exit status `code`.
    pub fn error(code: i32, message: impl Into<String>) -> Self {
        let message = message.into();
        crate::selphi_error!("{}", message);
        Fatal { code, message }
    }

    /// Report `message` verbatim on stderr (not in the `.log` file) and return it
    /// with exit status `code`.
    pub fn stderr(code: i32, message: impl Into<String>) -> Self {
        let message = message.into();
        eprintln!("{}", message);
        Fatal { code, message }
    }

    /// The exit status the command line ends with.
    pub fn code(&self) -> i32 { self.code }

    /// What went wrong, as it was reported.
    pub fn message(&self) -> &str { &self.message }

    /// The [`Fatal`] an `io::Error` carries, if it carries one.
    pub fn of(e: &io::Error) -> Option<&Fatal> {
        e.get_ref().and_then(|inner| inner.downcast_ref::<Fatal>())
    }
}

impl fmt::Display for Fatal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.message) }
}

impl std::error::Error for Fatal {}

impl From<Fatal> for io::Error {
    fn from(f: Fatal) -> Self { io::Error::other(f) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fatal_survives_an_io_error_and_keeps_its_code() {
        let e: io::Error = Fatal { code: 2, message: "refused".into() }.into();
        let f = Fatal::of(&e).expect("carried");
        assert_eq!((f.code(), f.message()), (2, "refused"));
        assert_eq!(e.to_string(), "refused");
        assert!(Fatal::of(&io::Error::new(io::ErrorKind::NotFound, "gone")).is_none());
    }

    #[test]
    fn library_refusals_return_the_status_the_command_line_exits_with() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.vcf.gz");
        let e = crate::io::target_io::read_cohort_vcf(missing.to_str().unwrap()).unwrap_err();
        let f = Fatal::of(&e).expect("a reported refusal");
        assert_eq!(f.code(), 1);
        assert!(f.message().starts_with(&format!("Cannot open {}: ", missing.display())), "{}", f.message());
        let same = dir.path().join("x.vcf.gz");
        std::fs::write(&same, b"").unwrap();
        let f = crate::common::utils::refuse_overwrite(&[same.to_str().unwrap()], std::slice::from_ref(&same)).unwrap_err();
        assert_eq!(f.code(), 2);
        assert!(crate::io::variant_only::load_callable(Some(missing.to_str().unwrap())).is_err_and(|f| f.code() == 2));
        assert_eq!(crate::io::target_io::refuse_uncalled(3).code(), 1);
    }
}
