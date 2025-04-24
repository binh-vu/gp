use crate::error::GramsError;
use hashbrown::HashMap;
use indicatif::ProgressStyle;
use postcard::to_allocvec;
use rayon::prelude::ParallelIterator;
use std::borrow::Borrow;
use std::cmp::Eq;
use std::hash::Hash;
use std::io::{Read, Write};
use std::ops::Index;

pub struct ByValue;
pub struct ByReference;

pub trait ReturnKind<'a, T: Sized + 'a> {
    type Type: Sized;
}

impl<'a, T: Sized + 'a> ReturnKind<'a, T> for ByValue {
    type Type = T;
}

impl<'a, T: Sized + 'a> ReturnKind<'a, T> for ByReference {
    type Type = &'a T;
}

/// A map that allow to access a node by both position and key
pub struct PositionMap<K: Eq + Hash, V> {
    data: Vec<V>,
    index: HashMap<K, usize>,
}

impl<K: Eq + Hash, V> PositionMap<K, V> {
    pub fn insert(&mut self, key: K, value: V) {
        let index = self.data.len();
        self.data.push(value);
        self.index.insert(key, index);
    }

    pub fn get_pos<Q: ?Sized>(&self, k: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.index.get(k).cloned()
    }

    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.index.get(k).map(|&index| &self.data[index])
    }

    pub fn get_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.index.get(k).map(|&index| &mut self.data[index])
    }
}

impl<K: Eq + Hash, V> Index<usize> for PositionMap<K, V> {
    type Output = V;

    fn index(&self, index: usize) -> &V {
        &self.data[index]
    }
}

#[derive(PartialEq, PartialOrd, Debug, Clone, Copy)]
pub struct OrdF64(pub f64);

impl Eq for OrdF64 {}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub fn save_object(obj: &impl serde::Serialize, outfile: &str) -> Result<(), GramsError> {
    let mut file = std::fs::File::create(outfile)?;
    file.write_all(&to_allocvec(obj)?)?;
    Ok(())
}

pub fn load_object<T: serde::de::DeserializeOwned>(outfile: &str) -> Result<T, GramsError> {
    let mut file = std::fs::File::open(outfile)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(postcard::from_bytes(&buffer)?)
}

pub fn get_progress_bar_style(tag: &str) -> ProgressStyle {
    ProgressStyle::with_template(&format!(
        "{}: {}",
        tag, "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}"
    ))
    .unwrap()
}

pub enum IteratorKind<I1, I2, Item>
where
    I1: ParallelIterator<Item = Item>,
    I2: Iterator<Item = Item>,
    Item: Sync + Send,
{
    Parallel(I1),
    Sequential(I2),
}
