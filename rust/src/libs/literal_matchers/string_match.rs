use crate::{error::GramsError, models::AlgoContext};
use anyhow::Result;
use kgdata_core::models::Value;

use super::{parsed_text_repr::ParsedTextRepr, SingleTypeMatcher};

pub struct StringExactTest;

impl SingleTypeMatcher for StringExactTest {
    fn get_name(&self) -> &'static str {
        "string_exact_test"
    }

    fn compare(
        &self,
        query: &ParsedTextRepr,
        key: &Value,
        _context: &AlgoContext,
    ) -> Result<(bool, f64), GramsError> {
        if key.as_string().unwrap().trim() == &query.normed_string {
            Ok((true, 1.0))
        } else {
            Ok((false, 0.0))
        }
    }
}
