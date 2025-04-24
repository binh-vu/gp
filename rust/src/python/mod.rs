pub mod algorithms;
pub mod models;

pub mod literal_matchers {
    use pyo3::prelude::*;

    use crate::libs::literal_matchers::parsed_text_repr::{ParsedDatetimeRepr, ParsedNumberRepr};
    use crate::libs::literal_matchers::{LiteralMatcherConfig, ParsedTextRepr, PyLiteralMatcher};

    pub(crate) fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
        let submodule = PyModule::new(py, "literal_matchers")?;

        m.add_submodule(submodule)?;

        submodule.add_class::<LiteralMatcherConfig>()?;
        submodule.add_class::<PyLiteralMatcher>()?;
        submodule.add_class::<ParsedTextRepr>()?;
        submodule.add_class::<ParsedNumberRepr>()?;
        submodule.add_class::<ParsedDatetimeRepr>()?;

        py.import("sys")?
            .getattr("modules")?
            .set_item("gp_core.literal_matchers", submodule)?;

        Ok(())
    }
}
