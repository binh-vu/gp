pub mod candidate_local_search;
pub mod data_matching;
pub mod extract_can_graph_feature;
pub mod extract_entity_linking_feature;

use pyo3::prelude::*;

pub(crate) fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "algorithms")?;
    m.add_submodule(submodule)?;

    submodule.add_class::<self::candidate_local_search::CandidateLocalSearchConfig>()?;
    submodule.add_function(wrap_pyfunction!(
        self::candidate_local_search::py_candidate_local_search,
        submodule
    )?)?;
    submodule.add_function(wrap_pyfunction!(
        self::candidate_local_search::py_par_candidate_local_search,
        submodule
    )?)?;

    submodule.add_function(wrap_pyfunction!(self::data_matching::matching, submodule)?)?;
    submodule.add_function(wrap_pyfunction!(
        self::data_matching::par_matching,
        submodule
    )?)?;
    submodule.add_class::<self::data_matching::PyDataMatchesResult>()?;
    submodule.add_class::<self::data_matching::potential_relationships_list_view::ListView>()?;
    submodule.add_class::<self::data_matching::matched_ent_rel_list_view::ListView>()?;
    submodule.add_class::<self::data_matching::matched_statement_list_view::ListView>()?;
    submodule.add_class::<self::data_matching::matched_qualifier_list_view::ListView>()?;
    submodule.add_class::<self::data_matching::PotentialRelationshipsView>()?;
    submodule.add_class::<self::data_matching::MatchedEntRelView>()?;
    submodule.add_class::<self::data_matching::MatchedStatementView>()?;
    submodule.add_class::<self::data_matching::MatchedQualifierView>()?;

    submodule.add_function(wrap_pyfunction!(
        self::extract_can_graph_feature::py_extract_cangraph,
        submodule
    )?)?;
    submodule.add_function(wrap_pyfunction!(
        self::extract_can_graph_feature::py_par_extract_cangraph,
        submodule
    )?)?;
    submodule.add_class::<self::extract_can_graph_feature::CanGraphExtractorCfg>()?;
    submodule.add_class::<self::extract_can_graph_feature::PyCanGraphExtractedResult>()?;

    submodule.add_function(wrap_pyfunction!(
        self::extract_entity_linking_feature::par_extract_candidate_entity_link_freqs,
        submodule
    )?)?;
    submodule.add_function(wrap_pyfunction!(
        self::extract_entity_linking_feature::extract_candidate_entity_link_freqs,
        submodule
    )?)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("gp_core.algorithms", submodule)?;

    Ok(())
}
