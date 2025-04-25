// #![allow(warnings, unused)]

// use anyhow::Result;
// use gp::{
//     error::GramsError,
//     models::{db::RemoteGramsDB, AlgoContext},
//     python::algorithms::extract_can_graph_feature::py_par_extract_cangraph,
// };
// #[allow(unused_imports)]
// use gp::{
//     helper::load_object,
//     libs::literal_matchers::{LiteralMatcher, LiteralMatcherConfig, PyLiteralMatcher},
//     models::{LocalGramsDB, Table, TableCells},
//     python::{
//         algorithms::extract_can_graph_feature::{self, extract_cangraph, CanGraphExtractorCfg},
//         models::PyAlgoContext,
//     },
// };
// #[allow(unused_imports)]
// use hashbrown::HashSet;
// #[allow(unused_imports)]
// use indicatif::{ParallelProgressIterator, ProgressIterator};
// use itertools::Itertools;
// use kgdata_core::db::Map;
// use once_cell::sync::Lazy;
// #[allow(unused_imports)]
// use pyo3::prelude::*;
// use rand::seq::SliceRandom;
// use rand::thread_rng;
// use rayon::{current_thread_index, prelude::*};
// #[allow(unused_imports)]
// use std::time::{Duration, Instant};

// static basedir: &str = "/var/tmp/ben/ray-on-100";
// static BENCH_READ_DIR: &str = "/dev/shm/disk-throttle";

// fn get_cpus() -> usize {
//     std::env::var("N_CPUS")
//         .unwrap_or(num_cpus::get().to_string())
//         .as_str()
//         .parse::<usize>()
//         .unwrap()
// }

// const SOCKET_BASE_URL: &str = "ipc:///dev/shm/kgdata";
// fn get_datadir() -> String {
//     format!("{}/kgdata/databases/wikidata/20230619", "/var/tmp")
// }

// fn get_db() -> Result<LocalGramsDB> {
//     Ok(LocalGramsDB::new(&get_datadir())?)
// }

// fn get_n_entdb() -> usize {
//     std::env::var("N_ENTDB")
//         .unwrap_or(num_cpus::get().to_string())
//         .as_str()
//         .parse::<usize>()
//         .unwrap()
// }

// fn get_remote_db() -> RemoteGramsDB {
//     let n_entdb = get_n_entdb();
//     let n_entmetadb = std::env::var("N_ENTMETADB")
//         .unwrap_or(num_cpus::get().to_string())
//         .as_str()
//         .parse::<usize>()
//         .unwrap();

//     let entity_urls = (0..n_entdb)
//         .into_iter()
//         // .map(|i| format!("{}/entity.{:0>3}.ipc", SOCKET_BASE_URL, i))
//         .map(|i| format!("tcp://ckg03.isi.edu:{}", 35500 + i))
//         .collect::<Vec<_>>();
//     let entity_metadata_urls = (0..n_entmetadb)
//         .into_iter()
//         // .map(|i| format!("{}/entity_metadata.{:0>3}.ipc", SOCKET_BASE_URL, i))
//         .map(|i| format!("tcp://ckg03.isi.edu:{}", 35600 + i))
//         .collect::<Vec<_>>();
//     RemoteGramsDB::new(&get_datadir(), &entity_urls, &entity_metadata_urls, 64, 128).unwrap()
// }

#[allow(unused_variables)]
fn main() {
    //     procspawn::init();

    //     let args: Vec<String> = std::env::args().collect();
    //     println!("args: {:?}", args);

    //     // let DB_DIR = &args[1];
    //     // let bench_read_dir = &args[2];
    //     let MODE = &args[1];

    //     let mut start = Instant::now();
    //     let mut cfg = load_object::<CanGraphExtractorCfg>(&format!("{}/cangraph.cfg.bin", basedir))?;
    //     cfg.run_subproperty_inference = false;
    //     cfg.run_transitive_inference = false;

    //     let mut tables_and_cells = (0..250)
    //         .into_iter()
    //         .map(|i| {
    //             let table = load_object::<Table>(&format!("{}/{}/table.bin", basedir, i)).unwrap();
    //             let cells = load_object::<TableCells>(&format!("{}/{}/cells.bin", basedir, i)).unwrap();
    //             (table, cells)
    //         })
    //         // .filter(|(table, _cells)| table.n_rows() < 100)
    //         .collect::<Vec<_>>();

    //     // tables_and_cells.truncate(50);

    //     let mut entlist = (0..250)
    //         .into_iter()
    //         .map(|i| {
    //             let filename = format!("{}/{}/entids.txt", basedir, i);
    //             std::fs::read_to_string(&filename)
    //                 .unwrap()
    //                 .lines()
    //                 .filter(|x| x.len() > 0)
    //                 .map(String::from)
    //                 .collect::<Vec<_>>()
    //         })
    //         // .map(|x| x[0..1000].to_vec())
    //         .collect::<Vec<_>>();
    //     // entlist.truncate(50);

    //     println!("Deser data takes: {:?}", start.elapsed());

    //     // start = Instant::now();
    //     // let db = get_db()?;
    //     // println!("Load DB takes: {:?}", start.elapsed());

    //     start = Instant::now();

    //     //     // write_ent_list_par(&entlist);
    //     // if MODE == "thread" {
    //     //     fetch_ent_list_thread_par(&entlist);
    //     // }
    //     // if MODE == "proc" {
    //     //     fetch_ent_list_proc_par(&entlist);
    //     // }
    //     // if args[1] == "thread" {

    //     // }
    //     // if args[1] == "proc" {

    //     // }
    //     //     // fetch_ent_list_thread_par(entlist);
    //     //     // fetch_entities_consecutive(&db, &tables_and_cells);
    //     //     // fetch_entities_nested_par(&db, &tables_and_cells);
    //     //     // fetch_entities_par(&db, &tables_and_cells);
    //     //     // fetch_entities_chunk_par(&tables_and_cells);

    //     if MODE == "proc" {
    //         let algocontexts = extract_algo_context_proc_par(&cfg, &tables_and_cells)?;
    //     }
    //     if MODE == "thread" {
    //         let algocontexts = extract_algo_context_par(&cfg, &tables_and_cells)?;
    //     }
    //     //     // let algocontexts = extract_algo_context_nested_par(&db, &cfg, &tables_and_cells)?;

    //     println!("Main bench takes: {:?}", start.elapsed());
    //     //     // extract_cangraph_par(&db, &cfg, &tables_and_cells);
    //     //     // Python::with_gil(|py| {
    //     //     //     extract_cangraph_par_py(py, &db, &cfg, tables_and_cells);
    //     //     // });
    //     //     // extract_cangraph_precontext_par(&db, &cfg, &tables_and_cells, algocontexts);

    // Ok(())
}

// // fn write_ent_list_par(entlist: &Vec<Vec<String>>) {
// //     let mut start = Instant::now();
// //     let db = get_db().unwrap();
// //     println!("Load DB takes: {:?}", start.elapsed());

// //     start = Instant::now();
// //     entlist
// //         .into_par_iter()
// //         .enumerate()
// //         .progress()
// //         .for_each(|(ei, entids)| {
// //             let mut content = db
// //                 .par_fetch_entities(&entids)
// //                 .unwrap()
// //                 .into_values()
// //                 .map(|ent| serde_json::to_string(&ent).unwrap())
// //                 .collect::<Vec<_>>();

// //             content.shuffle(&mut thread_rng());
// //             let mut newcontent = Vec::with_capacity(50000);
// //             for i in 0..50000 {
// //                 newcontent.push(content[i % content.len()].as_str());
// //             }

// //             std::fs::write(
// //                 &format!("{}/{:0>3}.jl", BENCH_READ_DIR, ei),
// //                 newcontent.join("\n"),
// //             )
// //             .unwrap()
// //         });
// //     println!("Write entities in parallel takes: {:?}", start.elapsed());
// // }

// // fn read_ents_par(bench_read_dir: &str, n_files: usize) {
// //     let start = Instant::now();
// //     let res = (0..n_files)
// //         .into_par_iter()
// //         .progress()
// //         .map(|ei| {
// //             std::fs::read_to_string(&format!("{}/{:0>3}.jl", bench_read_dir, ei))
// //                 .expect(&format!("{}/{:0>3}.jl", bench_read_dir, ei))
// //                 .len()
// //             // .split("\n")
// //             // .map(|line| serde_json::from_str::<kgdata_core::models::Entity>(&line).unwrap())
// //             // .map(|ent| ent.id[1..].parse::<usize>().unwrap() % 2)
// //             // .map(|line| line.len() as usize)
// //             // .collect::<Vec<_>>()
// //         })
// //         .collect::<Vec<_>>();
// //     println!("Read files in parallel takes: {:?}", start.elapsed());
// //     println!("Result: {}", res.into_iter().sum::<usize>())
// // }

// // fn read_ents_proc_par(bench_read_dir: &str, n_files: usize) {
// //     let start = Instant::now();
// //     let chunks = get_cpus();
// //     let jobs = (0..chunks)
// //         .into_iter()
// //         .map(|idx| {
// //             let args = (0..n_files)
// //                 .into_iter()
// //                 .filter(|i| i % chunks == idx)
// //                 .map(|ei| format!("{}/{:0>3}.jl", bench_read_dir, ei))
// //                 .collect::<Vec<_>>();

// //             procspawn::spawn(args, |args: Vec<String>| {
// //                 args.into_iter()
// //                     .map(|filename| {
// //                         std::fs::read_to_string(&filename).unwrap().len()
// //                         // .split("\n")
// //                         // // .map(|line| {
// //                         // //     serde_json::from_str::<kgdata_core::models::Entity>(&line).unwrap()
// //                         // // })
// //                         // // .map(|ent| ent.id[1..].parse::<usize>().unwrap() % 2)
// //                         // .map(|line| line.len() as usize)
// //                         // .collect::<Vec<_>>()
// //                     })
// //                     // .flatten()
// //                     .collect::<Vec<_>>()
// //             })
// //         })
// //         .progress()
// //         .collect::<Vec<_>>();

// //     let res = jobs
// //         .into_iter()
// //         .map(|job| job.join().unwrap())
// //         .progress()
// //         .flatten()
// //         .collect::<Vec<_>>();
// //     println!("Fetch files in proc parallel takes: {:?}", start.elapsed());
// //     println!("Result: {}", res.into_iter().sum::<usize>());
// // }

// // fn fetch_ent_list_par(entlist: &Vec<Vec<String>>) {
// //     let mut start = Instant::now();
// //     let db = get_db().unwrap();
// //     println!("Load DB takes: {:?}", start.elapsed());

// //     start = Instant::now();
// //     let res = entlist
// //         .into_par_iter()
// //         .map(|entids| {
// //             db.par_fetch_entities(&entids)
// //                 .unwrap()
// //                 .into_keys()
// //                 .map(|id| id[1..].parse::<i32>().unwrap() % 2)
// //                 .collect::<Vec<_>>()
// //         })
// //         .progress()
// //         .collect::<Vec<_>>();
// //     println!("Fetch entities in parallel takes: {:?}", start.elapsed());
// //     println!("Result: {}", res.into_iter().flatten().sum::<i32>())
// // }

// fn fetch_ent_list_thread_par(entlist: &Vec<Vec<String>>) {
//     let db = get_remote_db();

//     let mut start = Instant::now();
//     let res = entlist
//         .into_par_iter()
//         .map(|entids| {
//             db.0.entities
//                 .par_slice_get_exist_as_map(entids)
//                 .unwrap()
//                 .into_keys()
//                 .map(|id| id[1..].parse::<i32>().unwrap() % 2)
//                 .collect::<Vec<_>>()
//         })
//         .progress()
//         .flatten()
//         .collect::<Vec<_>>();

//     println!(
//         "Fetch entities in thread parallel takes: {:?}",
//         start.elapsed()
//     );
//     println!("Result: {}", res.into_iter().sum::<i32>())
// }

// fn fetch_ent_list_proc_par(entlist: &Vec<Vec<String>>) {
//     let mut start = Instant::now();
//     let chunks = get_cpus();

//     let jobs = (0..chunks)
//         .into_iter()
//         .map(|idx| {
//             let args = entlist
//                 .iter()
//                 .enumerate()
//                 .filter(|(i, _)| i % chunks == idx)
//                 .map(|(_, x)| x.clone())
//                 .collect::<Vec<_>>();

//             procspawn::spawn(args, |args: Vec<Vec<String>>| {
//                 let db = get_db().unwrap();
//                 args.into_iter()
//                     .map(|arg| {
//                         db.0.entities
//                             .slice_get_exist_as_map(&arg)
//                             .unwrap()
//                             .into_keys()
//                             .map(|id| id[1..].parse::<i32>().unwrap() % 2)
//                             .collect::<Vec<_>>()
//                     })
//                     .flatten()
//                     .collect::<Vec<_>>()
//             })
//         })
//         .progress()
//         .collect::<Vec<_>>();

//     let res = jobs
//         .into_iter()
//         .map(|job| job.join().unwrap())
//         .progress()
//         .flatten()
//         .collect::<Vec<_>>();
//     println!(
//         "Fetch entities in proc parallel takes: {:?}",
//         start.elapsed()
//     );
//     println!("Result: {}", res.into_iter().sum::<i32>())
// }

// // // fn fetch_entities_consecutive(db: &GramsDB, tables_and_cells: &Vec<(Table, TableCells)>) {
// // //     let start = Instant::now();
// // //     let mut res = vec![];
// // //     for (table, cells) in tables_and_cells.iter().progress() {
// // //         let ents =
// // //             db.0.entities
// // //                 .batch_get_exist_as_map(&db.get_table_entity_ids(table))
// // //                 .unwrap();
// // //         let entmeta = db.get_entities_metadata(&ents).unwrap();
// // //         res.push((ents, entmeta));
// // //     }
// // //     println!("Fetch entities takes: {:?}", start.elapsed());
// // // }

// // fn fetch_entities_nested_par(db: &GramsDB, tables_and_cells: &Vec<(Table, TableCells)>) {
// //     let start = Instant::now();
// //     let _res = (&tables_and_cells)
// //         .into_par_iter()
// //         .map(|(table, _cells)| {
// //             let entids = db.get_table_entity_ids(table);
// //             let ents = db.par_fetch_entities(&entids).unwrap();
// //             // let metaids = db.retrieve_unfetch_neighbor_entity_ids(&ents).unwrap();
// //             let metaids = db.retrieve_unfetch_neighbor_entity_ids(&entids, &ents, true);
// //             let entmeta = db
// //                 .par_fetch_entities_metadata(&metaids.into_iter().collect::<Vec<_>>())
// //                 .unwrap();
// //             // (ents, entmeta)
// //             (ents.len(), entmeta.len())
// //             // db.par_get_entities(entity_ids, n_hop)
// //             // ents
// //         })
// //         .progress()
// //         .collect::<Vec<_>>();
// //     println!(
// //         "Fetch entities in nested parallel takes: {:?}",
// //         start.elapsed()
// //     );
// // }

// // // fn fetch_entities_par(db: &GramsDB, tables_and_cells: &Vec<(Table, TableCells)>) {
// // //     let start = Instant::now();
// // //     let _res = (&tables_and_cells)
// // //         .into_par_iter()
// // //         .map(|(table, _cells)| {
// // //             let ents =
// // //                 db.0.entities
// // //                     .batch_get_exist_as_map(&db.get_table_entity_ids(table))
// // //                     .unwrap();
// // //             let entmeta = db.get_entities_metadata(&ents).unwrap();

// // //             // (ents, entmeta)
// // //             (ents.len(), entmeta.len())
// // //             // ents
// // //         })
// // //         .progress()
// // //         .collect::<Vec<_>>();
// // //     println!("Fetch entities in parallel takes: {:?}", start.elapsed());
// // // }

// // // fn fetch_entities_chunk_par(tables_and_cells: &Vec<(Table, TableCells)>) {
// // //     let start = Instant::now();
// // //     let chunks = get_cpus();

// // //     let res = (0..chunks)
// // //         .into_par_iter()
// // //         .map(|idx| {
// // //             println!("Thread number: {:?}", current_thread_index());
// // //             let db = get_db().unwrap();
// // //             let mut out = Vec::new();
// // //             for (i, (table, _cells)) in tables_and_cells.iter().enumerate() {
// // //                 if i % chunks == idx {
// // //                     let ents =
// // //                         db.0.entities
// // //                             .batch_get_exist_as_map(&db.get_table_entity_ids(table))
// // //                             .unwrap();
// // //                     let entmetadata = db.get_entities_metadata(&ents).unwrap();

// // //                     out.push((ents, entmetadata));
// // //                 }
// // //             }
// // //             out.len()
// // //         })
// // //         .progress()
// // //         .collect::<Vec<_>>();

// // //     println!(
// // //         "Fetch entities in chunk parallel takes: {:?}",
// // //         start.elapsed()
// // //     );
// // // }

// fn extract_algo_context_proc_par(
//     cfg: &CanGraphExtractorCfg,
//     tables_and_cells: &Vec<(Table, TableCells)>,
// ) -> Result<Vec<usize>, GramsError> {
//     let start = Instant::now();
//     let mut res = vec![];

//     let chunks = get_cpus();

//     let jobs = (0..chunks)
//         .into_iter()
//         .map(|idx| {
//             let tables = tables_and_cells
//                 .iter()
//                 .enumerate()
//                 .filter(|(i, _)| i % chunks == idx)
//                 .map(|(_, (table, _cells))| table.clone())
//                 .collect::<Vec<_>>();

//             procspawn::spawn(tables, extract_algo_contexts_proc_par__fn)
//         })
//         .progress()
//         .collect::<Vec<_>>();

//     res = jobs
//         .into_iter()
//         .map(|job| job.join().unwrap())
//         .progress()
//         .flatten()
//         .collect::<Vec<_>>();

//     // println!("starting pool...");
//     // let poolsize = 10;
//     // let pool = procspawn::Pool::new(poolsize).unwrap();
//     // let mut jobs = Vec::with_capacity(poolsize);
//     // // let mut unfinished = Vec::with_capacity(poolsize);
//     // let mut finished = Vec::with_capacity(poolsize);

//     // println!("starting pool done...");
//     // // for table, _cells in tables_and_cells {
//     // //     let job = pool.spawn(table.clone(), extract_algo_context_proc_par__fn);
//     // //     if jobs.len() == poolsize {
//     // //         // wait for one job to finish before adding another...

//     // //     }
//     // // }
//     // for (table, _cells) in tables_and_cells.into_iter().progress() {
//     //     let newjob = pool.spawn(table.clone(), extract_algo_context_proc_par__fn);
//     //     // println!("new job: {:?}", newjob.join());
//     //     jobs.push(newjob);

//     //     if jobs.len() >= 1 {
//     //         print!("checking ");
//     //         loop {
//     //             for (i, job) in jobs.iter_mut().enumerate() {
//     //                 let jobres = job.join_timeout(Duration::from_millis(200));
//     //                 println!("{:?}", jobres);
//     //                 print!(".");
//     //                 let jobres = job.join_timeout(Duration::from_millis(200));
//     //                 println!("{:?}", jobres);
//     //                 print!(".");

//     //                 // match jobres {
//     //                 //     Ok(r) => {
//     //                 //         res.push(r);
//     //                 //         finished.push(i);
//     //                 //     }
//     //                 //     Err(e) => {
//     //                 //         if e.is_timeout() {
//     //                 //             continue;
//     //                 //         } else if e.is_panic() {
//     //                 //             panic!(
//     //                 //                 "error: {:?}",
//     //                 //                 e.panic_info().expect("get panic error").message()
//     //                 //             );
//     //                 //         } else {
//     //                 //             panic!("error: {:?}", e);
//     //                 //         }
//     //                 //     }
//     //                 // }
//     //             }
//     //             print!("_");

//     //             if finished.len() > 0 {
//     //                 for i in finished.drain(..).rev() {
//     //                     jobs.remove(i);
//     //                 }
//     //                 print!(" done! {}", jobs.len());
//     //                 break;
//     //             }
//     //         }
//     //     }
//     // }
//     // println!("submit all jobs");
//     // for job in jobs.into_iter().progress() {
//     //     res.push(job.join().unwrap());
//     // }

//     // pool.shutdown();
//     println!(
//         "Extract algo context in process parallel takes: {:?}",
//         start.elapsed()
//     );

//     Ok(res)
// }

// fn extract_algo_context_par(
//     cfg: &CanGraphExtractorCfg,
//     tables_and_cells: &Vec<(Table, TableCells)>,
// ) -> Result<Vec<usize>, GramsError> {
//     let db = get_remote_db();

//     let start = Instant::now();
//     let res = tables_and_cells
//         .into_par_iter()
//         .with_max_len(1)
//         .map(|(table, _cells)| Ok(db.get_algo_context(table, cfg.n_hop, true)?.entities.len()))
//         .progress()
//         .collect::<Result<Vec<_>, GramsError>>()?;
//     println!(
//         "Extract algo context in parallel takes: {:?}",
//         start.elapsed()
//     );
//     Ok(res)
// }

// // fn extract_algo_context_nested_par(
// //     db: &GramsDB,
// //     cfg: &CanGraphExtractorCfg,
// //     tables_and_cells: &Vec<(Table, TableCells)>,
// // ) -> Result<Vec<AlgoContext>, GramsError> {
// //     let start = Instant::now();
// //     let res = tables_and_cells
// //         .into_par_iter()
// //         .map(|(table, _cells)| db.par_get_algo_context(table, cfg.n_hop))
// //         .progress()
// //         .collect::<Result<Vec<_>, GramsError>>();
// //     println!(
// //         "Extract algo context in nested parallel takes: {:?}",
// //         start.elapsed()
// //     );
// //     res
// // }

// // fn extract_cangraph_precontext_par(
// //     db: &PyGramsDB,
// //     cfg: &CanGraphExtractorCfg,
// //     tables_and_cells: &Vec<(Table, TableCells)>,
// //     context: Vec<AlgoContext>,
// // ) {
// //     let start = Instant::now();
// //     let res = tables_and_cells
// //         .into_par_iter()
// //         .progress()
// //         .zip(context.into_par_iter())
// //         .map(|((table, cells), mut context)| {
// //             extract_cangraph(table, cells, db, Some(&mut context), cfg)
// //         })
// //         .collect::<Result<Vec<_>, GramsError>>();
// //     println!("Extract can graph in parallel takes: {:?}", start.elapsed());
// //     res.unwrap();
// // }

// // fn extract_cangraph_par(
// //     db: &PyGramsDB,
// //     cfg: &CanGraphExtractorCfg,
// //     tables_and_cells: &Vec<(Table, TableCells)>,
// // ) {
// //     let start = Instant::now();
// //     let res = tables_and_cells
// //         .into_par_iter()
// //         .progress()
// //         .map(|(table, cells)| extract_cangraph(table, cells, db, None, cfg))
// //         .collect::<Result<Vec<_>, GramsError>>();
// //     println!("Extract can graph in parallel takes: {:?}", start.elapsed());
// //     res.unwrap();
// // }

// // fn extract_cangraph_par_py(
// //     py: Python<'_>,
// //     db: &PyGramsDB,
// //     cfg: &CanGraphExtractorCfg,
// //     tables_and_cells: Vec<(Table, TableCells)>,
// // ) {
// //     let start = Instant::now();
// //     let mut tables = Vec::with_capacity(tables_and_cells.len());
// //     let mut cells = Vec::with_capacity(tables_and_cells.len());

// //     for (tbl, cell) in tables_and_cells {
// //         tables.push(Py::new(py, tbl).unwrap());
// //         cells.push(Py::new(py, cell).unwrap());
// //     }

// //     let res = py_par_extract_cangraph(py, tables, cells, db, cfg, None, true);
// //     println!(
// //         "Extract can graph in python parallel takes: {:?}",
// //         start.elapsed()
// //     );
// //     res.unwrap();
// // }

// // fn extract_algo_context_proc_par__fn(table: Table) -> usize {
// //     get_db()
// //         .unwrap()
// //         .get_algo_context(&table, 1)
// //         .unwrap()
// //         .entities
// //         .len()

// //     // unsafe {
// //     //     if DB.is_none() {
// //     //         DB = Some(get_db().unwrap());
// //     //     }
// //     //     DB.as_ref()
// //     //         .unwrap()
// //     //         .get_algo_context(&table, 1)
// //     //         .unwrap()
// //     //         .entities
// //     //         .len()
// //     // }
// // }

// fn extract_algo_contexts_proc_par__fn(tables: Vec<Table>) -> Vec<usize> {
//     let db = get_db().unwrap();
//     tables
//         .into_iter()
//         .map(|table| {
//             db.get_algo_context(&table, 1, false)
//                 .unwrap()
//                 .entities
//                 .len()
//         })
//         .collect::<Vec<_>>()
// }
