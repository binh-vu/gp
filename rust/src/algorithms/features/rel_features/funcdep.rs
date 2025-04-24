use hashbrown::{HashMap, HashSet};

use crate::models::{Table, TableCells};

use super::{
    super::feature_store::{FeatureStore, FeatureStoreCache},
    TargetRelationship,
};

pub trait FunctionalDependencyDetector {
    /// Test whether values in the target column is uniquely determined by the values in the source column.
    /// True if it's FD.
    fn is_functional_dependency(
        &mut self,
        source_column_index: usize,
        target_column_index: usize,
    ) -> bool;
}

/// for each rel, test if the edge of the relationship does not connect two columns
/// which form a functional dependency (item in target is uniquely determined by item in source)
pub fn not_func_dependency<D: FunctionalDependencyDetector>(
    store: &FeatureStore,
    _storecache: &mut FeatureStoreCache,
    func_dep_detector: &mut D,
    rels: &[TargetRelationship],
) -> Vec<i32> {
    let mut notfuncdep: HashMap<(usize, usize), i32> = HashMap::new();
    rels.iter()
        .map(|rel| {
            let u = store.cg.graph.get_node(rel.source.source).unwrap();
            let v = store.cg.graph.get_node(rel.target.target).unwrap();

            if !u.is_column() || !v.is_column() {
                return -1;
            }

            let ucol = u.try_as_column().unwrap().column;
            let vcol = v.try_as_column().unwrap().column;

            if !notfuncdep.contains_key(&(ucol, vcol)) {
                notfuncdep.insert(
                    (ucol, vcol),
                    (!func_dep_detector.is_functional_dependency(ucol, vcol)) as i32,
                );
            }

            *notfuncdep.get(&(ucol, vcol)).unwrap()
        })
        .collect::<Vec<_>>()
}

pub struct SimpleFunctionalDependencyDetector<'t> {
    pub table: &'t Table,
    pub cells: &'t TableCells,
    pub column_maps: Vec<Option<HashMap<String, Vec<usize>>>>,
}

impl<'t> FunctionalDependencyDetector for SimpleFunctionalDependencyDetector<'t> {
    fn is_functional_dependency(
        &mut self,
        source_column_index: usize,
        target_column_index: usize,
    ) -> bool {
        self._compute_value_map(source_column_index);
        let source_map = self.column_maps[source_column_index].as_ref().unwrap();

        let mut n_violate_fd = 0;
        for (_key, rows) in source_map.iter() {
            let target_keys = rows
                .iter()
                .map(|&ri| &self.cells.0[ri][target_column_index].normed_string)
                .collect::<HashSet<_>>();

            if target_keys.len() > 1 {
                n_violate_fd += 1;
            }
        }

        if source_map.len() == 0 {
            return true;
        }
        if (n_violate_fd as f64 / source_map.len() as f64) > 0.01 {
            return false;
        }
        return true;
    }
}

impl<'t> SimpleFunctionalDependencyDetector<'t> {
    pub fn new(table: &'t Table, cells: &'t TableCells) -> Self {
        Self {
            table,
            cells,
            column_maps: vec![None; table.n_cols()],
        }
    }

    /// Get a map of values in a column to its row numbers (possible duplication).
    /// This function is not perfect now. The value of the column is considered to be list of entities (if exist) or
    /// just the value of the cell
    fn _compute_value_map(&mut self, column_index: usize) {
        if self.column_maps[column_index].is_none() {
            let mut map = HashMap::<String, Vec<usize>>::new();
            for (ri, row) in self.cells.0.iter().enumerate() {
                let cell = &row[column_index].normed_string;
                if map.contains_key(cell) {
                    map.get_mut(cell).unwrap().push(ri);
                } else {
                    map.insert(cell.clone(), vec![ri]);
                }
            }
            self.column_maps[column_index] = Some(map);
        }
    }
}
