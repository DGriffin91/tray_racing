use std::{collections::HashMap, error::Error, fs::File, path::Path, time::Instant};

use chrono::{DateTime, Utc};
use tabled::{settings::Style, Table, Tabled};

use crate::{render_from_options, seconds_to_hh_mm_ss, Options};

pub fn tune(init_options: Options, mut event_loop: winit::event_loop::EventLoop<()>) {
    let mut model_cache = if init_options.disable_auto_tune_model_cache {
        None
    } else {
        Some(HashMap::new())
    };
    let mut results = Vec::new();
    let splits = [false];
    let search_distances = [14];
    let sort_precisions = [64];
    let reinsertion_batch_ratios = [0.1];
    let search_depth_thresholds = [0];
    let max_prims_per_leaf = [1, 3, 6, 8, 12];
    let collapse_traversal_cost = [1.0, 2.0, 3.0, 4.0, 8.0, 12.0];
    let permutations = splits.len()
        * search_distances.len()
        * sort_precisions.len()
        * reinsertion_batch_ratios.len()
        * search_depth_thresholds.len()
        * max_prims_per_leaf.len()
        * collapse_traversal_cost.len();
    {
        // Warmup. If skipped the first permutation or so may be faster because the clock speed has not normalized.
        let (_, _, _) = render_from_options(
            &init_options,
            &mut event_loop,
            &mut model_cache,
            &mut Vec::new(),
        );
    }
    println!("Evaluating {} permutations", permutations);
    let test_start_time = Instant::now();
    let mut best_avg_traversal_time = f32::MAX;
    let mut best_avg_blas_build_time = f32::MAX;
    let mut best_avg_tlas_build_time = f32::MAX;
    for split in splits {
        for search_distance in search_distances {
            for sort_precision in sort_precisions {
                for reinsertion_batch_ratio in reinsertion_batch_ratios {
                    for search_depth_threshold in search_depth_thresholds {
                        for max_prims_per_leaf in max_prims_per_leaf {
                            for collapse_traversal_cost in collapse_traversal_cost {
                                let mut options = init_options.clone();

                                options.split = split;
                                options.search_distance = search_distance;
                                options.sort_precision = sort_precision;
                                options.reinsertion_batch_ratio = reinsertion_batch_ratio;
                                options.search_depth_threshold = search_depth_threshold;
                                options.max_prims_per_leaf = max_prims_per_leaf;
                                options.collapse_traversal_cost = collapse_traversal_cost;

                                let (avg_traversal_time, avg_blas_build_time, avg_tlas_build_time) =
                                    render_from_options(
                                        &options,
                                        &mut event_loop,
                                        &mut model_cache,
                                        &mut Vec::new(),
                                    );
                                best_avg_traversal_time =
                                    best_avg_traversal_time.min(avg_traversal_time);
                                best_avg_blas_build_time =
                                    best_avg_blas_build_time.min(avg_blas_build_time);
                                best_avg_tlas_build_time =
                                    best_avg_tlas_build_time.min(avg_tlas_build_time);

                                results.push(TuningSet {
                                    search_distance,
                                    sort_precision,
                                    reinsertion_batch_ratio,
                                    search_depth_threshold,
                                    avg_traversal_time,
                                    avg_blas_build_time,
                                    avg_tlas_build_time,
                                    split,
                                    max_prims_per_leaf,
                                    collapse_traversal_cost,
                                    norm_best_blas_build_time: 0.0,
                                    norm_best_tlas_build_time: 0.0,
                                    norm_best_traversal_time: 0.0,
                                });

                                let elapsed_time = test_start_time.elapsed().as_secs_f32();
                                let avg_permutation_duration =
                                    elapsed_time / (results.len() as f32);
                                let expected_remaining_duration = (permutations - results.len())
                                    as f32
                                    * avg_permutation_duration;

                                println!(
                                    "Expected Remaining Duration: {}",
                                    seconds_to_hh_mm_ss(expected_remaining_duration)
                                );
                                println!("Avg permutation time: {:.2}s", avg_permutation_duration);
                                println!("Time elapsed: {}", seconds_to_hh_mm_ss(elapsed_time));
                                println!("{} / {}", results.len(), permutations);
                            }
                        }
                    }
                }
            }
        }
    }

    for result in &mut results {
        result.norm_best_traversal_time = result.avg_traversal_time / best_avg_traversal_time;
        result.norm_best_blas_build_time = result.avg_blas_build_time / best_avg_blas_build_time;
        result.norm_best_tlas_build_time = result.avg_tlas_build_time / best_avg_tlas_build_time;
    }

    let mut blas_filtered_results = results.clone();
    let mut tlas_filtered_results = results.clone();

    for result in &results {
        // If there exists a result that has both a better traversal time and also a better build time than this one, then omit this one.
        blas_filtered_results.retain(|r| {
            !(result.avg_traversal_time < r.avg_traversal_time
                && result.avg_blas_build_time < r.avg_blas_build_time)
        });
        tlas_filtered_results.retain(|r| {
            !(result.avg_traversal_time < r.avg_traversal_time
                && result.avg_tlas_build_time < r.avg_tlas_build_time)
        });
    }

    println!(
        "{}",
        Table::new(&blas_filtered_results).with(Style::blank())
    );
    save_results("results", &results);
    save_results("blas_filtered_results", &blas_filtered_results);
    save_results("tlas_filtered_results", &tlas_filtered_results);

    fn save_results(name: &str, tlas_filtered_results: &[TuningSet]) {
        match save_tuning_results_to_csv(&tlas_filtered_results, name) {
            Ok(filename) => println!("CSV file saved successfully as '{}'.", filename),
            Err(e) => eprintln!("Error saving CSV file: {}", e),
        }
    }
}

#[derive(Debug, Tabled, Clone, Copy)]
struct TuningSet {
    /// Split large tris into multiple AABBs
    split: bool,
    /// In PLOC, the number of nodes before and after the current one that are evaluated for pairing
    search_distance: u32,
    /// Bits used for ploc radix sort
    sort_precision: u8,
    /// Typically 0..1: Ratio of nodes considered as candidates for reinsertion. Above 1 to evaluate the whole set multiple times
    reinsertion_batch_ratio: f32,
    /// Below this depth a search distance of 1 will be used
    search_depth_threshold: usize,
    /// Maximum primitives per leaf. For CWBVH the limit is 3
    max_prims_per_leaf: u32,
    /// Multiplier for traversal cost calculation during collapse. A higher value will result in more primitives per leaf.
    collapse_traversal_cost: f32,
    /// Average of the traversal times for all the scene for these settings
    avg_traversal_time: f32,
    /// Average of the builds times for all the scene for these settings
    avg_blas_build_time: f32,
    avg_tlas_build_time: f32,
    /// Normalized from best of all traversal times: worse than best is above 1
    norm_best_traversal_time: f32,
    /// Normalized from best of all build times: worse than best is above 1
    norm_best_blas_build_time: f32,
    norm_best_tlas_build_time: f32,
}

fn save_tuning_results_to_csv(
    tuning_sets: &[TuningSet],
    filename: &str,
) -> Result<String, Box<dyn Error>> {
    // Get the current date and time
    let now: DateTime<Utc> = Utc::now();
    let filename = format!("{}_{}.csv", filename, now.format("%Y-%m-%d_%H-%M-%S"));

    // Create a CSV writer
    let path = Path::new(&filename);
    let file = File::create(path)?;
    let mut wtr = csv::Writer::from_writer(file);

    // Write the headers
    wtr.write_record(&[
        "split",
        "search_distance",
        "sort_precision",
        "reinsertion_batch_ratio",
        "search_depth_threshold",
        "max_prims_per_leaf",
        "collapse_traversal_cost",
        "avg_traversal_time",
        "avg_blas_build_time",
        "avg_tlas_build_time",
        "norm_best_traversal_time",
        "norm_best_blas_build_time",
        "norm_best_tlas_build_time",
    ])?;

    // Write the data
    for tuning_set in tuning_sets {
        wtr.write_record(&[
            tuning_set.split.to_string(),
            tuning_set.search_distance.to_string(),
            tuning_set.sort_precision.to_string(),
            tuning_set.reinsertion_batch_ratio.to_string(),
            tuning_set.search_depth_threshold.to_string(),
            tuning_set.max_prims_per_leaf.to_string(),
            tuning_set.collapse_traversal_cost.to_string(),
            tuning_set.avg_traversal_time.to_string(),
            tuning_set.avg_blas_build_time.to_string(),
            tuning_set.avg_tlas_build_time.to_string(),
            tuning_set.norm_best_traversal_time.to_string(),
            tuning_set.norm_best_blas_build_time.to_string(),
            tuning_set.norm_best_tlas_build_time.to_string(),
        ])?;
    }

    // Flush the writer to ensure all data is written to the file
    wtr.flush()?;
    Ok(filename)
}
