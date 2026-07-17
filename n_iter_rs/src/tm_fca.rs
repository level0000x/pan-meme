//! Turing Machine → FCA Lattice mapping for BB(6) analysis.
//!
//! Implements the theoretical framework from B-24:
//! Maps a Turing machine's transition function to a formal context,
//! builds the FCA concept lattice, and applies the N-operator to
//! compute spectral radius ρ(J) as a halting/non-halting discriminant.

use crate::fca::{self, FcaLattice, FormalConcept};
use crate::n_operator::{DynamicsParams, IterResult};
use crate::pipeline::{self, LatticeStats};
use std::collections::{HashMap, HashSet};
use rand::Rng;

// ─── Turing Machine Representation ───────────────────────────────────────────

/// A single transition rule: (next_state, write_symbol, direction)
/// Direction: -1 = Left, 1 = Right, 0 = Halt
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transition {
    pub next_state: usize,   // 0..n_states, or n_states for HALT
    pub write: u8,           // 0 or 1
    pub direction: i8,       // -1, 1, or 0 (halt)
}

/// A Turing machine with n states, 2 symbols (0, 1).
#[derive(Debug, Clone)]
pub struct TuringMachine {
    pub name: String,
    pub n_states: usize,
    /// transitions[state][symbol] = Transition
    pub transitions: Vec<[Transition; 2]>,
    /// Standard format string (e.g. "1RB1RA_0LC1LE_...")
    pub std_format: String,
}

impl TuringMachine {
    /// Parse from standard bbchallenge format.
    ///
    /// Format: "XSGXSG_XSGXSG_..." where X∈{0,1} (write), S∈{L,R} (direction),
    /// G∈{A,B,C,D,E,F} (next state), and "---" or "1RZ" means halt.
    ///
    /// Example: "1RB1RA_0LC1LE_1LD1LC_1LA0LB_1LF1RE_---0RA"
    pub fn parse(name: &str, std_format: &str) -> Option<Self> {
        let parts: Vec<&str> = std_format.split('_').collect();
        if parts.is_empty() {
            return None;
        }
        let n_states = parts.len();
        if n_states > 26 {
            return None;
        }

        let state_to_idx: HashMap<char, usize> = ('A'..='Z')
            .take(n_states)
            .enumerate()
            .map(|(i, c)| (c, i))
            .collect();

        let mut transitions: Vec<[Transition; 2]> = Vec::with_capacity(n_states);

        for (si, part) in parts.iter().enumerate() {
            let mut row: [Transition; 2] = [
                Transition { next_state: 0, write: 0, direction: 0 },
                Transition { next_state: 0, write: 0, direction: 0 },
            ];

            for sym in 0..2 {
                let start = sym * 3;
                if start + 3 > part.len() {
                    return None;
                }
                let seg = &part[start..start + 3];

                if seg == "---" {
                    // Halt transition
                    row[sym] = Transition {
                        next_state: n_states, // HALT marker
                        write: 0,
                        direction: 0,
                    };
                } else {
                    let chars: Vec<char> = seg.chars().collect();
                    if chars.len() != 3 {
                        return None;
                    }

                    let write_char = chars[0];
                    let dir_char = chars[1];
                    let state_char = chars[2];

                    let write = if write_char == '0' { 0 } else { 1 };
                    let direction = match dir_char {
                        'L' => -1,
                        'R' => 1,
                        _ => return None,
                    };

                    let next_state = if state_char == 'Z' {
                        n_states // HALT
                    } else {
                        *state_to_idx.get(&state_char)?
                    };

                    row[sym] = Transition { next_state, write, direction };
                }
            }
            transitions.push(row);
        }

        // Validate: each transition to a non-halt state must be within range
        for (si, row) in transitions.iter().enumerate() {
            for sym in 0..2 {
                let t = &row[sym];
                if t.next_state != n_states && t.next_state >= n_states {
                    eprintln!("  Invalid transition: state {} symbol {} -> state {}",
                        si, sym, t.next_state);
                    return None;
                }
            }
        }

        Some(TuringMachine {
            name: name.to_string(),
            n_states,
            transitions,
            std_format: std_format.to_string(),
        })
    }

    /// Get all distinct transition types (q → q') present in this TM.
    pub fn transition_types(&self) -> Vec<(usize, usize)> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for si in 0..self.n_states {
            for sym in 0..2 {
                let t = &self.transitions[si][sym];
                if t.next_state < self.n_states {
                    // Non-halt transition
                    let key = (si, t.next_state);
                    if seen.insert(key) {
                        result.push(key);
                    }
                }
            }
        }
        result.sort();
        result
    }

    /// Simulate the TM for `max_steps` steps, returning the sequence of
    /// (state, head_position, tape_window) configurations.
    ///
    /// `window_l` determines how many cells to capture on each side of the head.
    /// Returns (configurations, halted, steps_taken).
    pub fn simulate(
        &self,
        max_steps: u64,
        window_l: usize,
    ) -> (Vec<(usize, isize, Vec<u8>)>, bool, u64) {
        let mut tape: Vec<u8> = vec![0; 1024];
        let mut head: isize = 512; // start in the middle
        let mut state: usize = 0; // start state A
        let mut configs: Vec<(usize, isize, Vec<u8>)> = Vec::new();

        // Record initial configuration
        let window = self.extract_window(&tape, head, window_l);
        configs.push((state, head, window));

        for _step in 0..max_steps {
            let symbol = tape[head as usize] as usize;
            let t = &self.transitions[state][symbol];

            if t.next_state == self.n_states {
                // Halted
                return (configs, true, _step + 1);
            }

            // Write
            tape[head as usize] = t.write;

            // Move
            head += t.direction as isize;

            // Expand tape if needed
            if head < 0 {
                let mut new_tape = vec![0u8; 1024];
                new_tape.extend(tape);
                tape = new_tape;
                head += 1024;
            } else if head as usize >= tape.len() {
                tape.extend(vec![0u8; 1024]);
            }

            // Next state
            state = t.next_state;

            // Record configuration
            let window = self.extract_window(&tape, head, window_l);
            configs.push((state, head, window));
        }

        (configs, false, max_steps)
    }

    /// Extract a window of size 2*window_l+1 around the head.
    fn extract_window(&self, tape: &[u8], head: isize, window_l: usize) -> Vec<u8> {
        let mut window = Vec::with_capacity(2 * window_l + 1);
        for offset in -(window_l as isize)..=(window_l as isize) {
            let pos = head + offset;
            if pos >= 0 && (pos as usize) < tape.len() {
                window.push(tape[pos as usize]);
            } else {
                window.push(0); // default to 0 for out-of-bounds
            }
        }
        window
    }
}

// ─── Formal Context Construction (per B-24 §2.1, revised) ────────────────────

/// Build the formal context from the TM's actual computation trace.
///
/// V2: Uses NO deduplication to create a rich lattice with many objects.
/// Adds global shared attributes (parity, density, state groups) to create
/// overlapping extents and prevent lattice degeneration.
///
/// Objects = sampled configurations from the trace (no dedup)
/// Attributes = position-symbol pairs + global shared attributes + state groups
pub fn build_trace_formal_context(
    tm: &TuringMachine,
    max_steps: u64,
    window_l: usize,
    sample_every: u64,
) -> (Vec<Vec<bool>>, Vec<String>, Vec<String>) {
    let n_positions = 2 * window_l + 1;
    let n_states = tm.n_states;
    let state_names: Vec<char> = ('A'..='Z').take(n_states).collect();

    // Step 1: Simulate the TM
    let (configs, halted, _steps) = tm.simulate(max_steps, window_l);

    // Step 2: Sample configurations WITHOUT deduplication
    let mut sampled: Vec<(usize, Vec<u8>)> = Vec::new();
    for (idx, (state, _head, window)) in configs.iter().enumerate() {
        if idx as u64 % sample_every != 0 {
            continue;
        }
        sampled.push((*state, window.clone()));
    }

    let n_objects = sampled.len();

    // ─── Attribute Construction ───────────────────────────────────────────

    // Layer 1: Sparse position-symbol attributes (binned to reduce sparsity)
    // Use "position mod 2" and "symbol" to create shared attributes
    let n_pos_attrs = 2 * 2; // pos_even_0, pos_even_1, pos_odd_0, pos_odd_1
    let pos_bin_attr = |pi: usize, sym: usize| -> usize {
        (pi % 2) * 2 + sym
    };

    // Layer 2: Global shared attributes (key innovation to prevent degeneration)
    let n_global_attrs = 8;
    // 0: window_has_1s (any 1 in window)
    // 1: window_all_0s
    // 2: window_ones_lt_25pct
    // 3: window_ones_25_50pct
    // 4: window_ones_50_75pct
    // 5: window_ones_gt_75pct
    // 6: head_at_center (head position = center of window)
    // 7: head_at_edge (head position at window edge)

    // Layer 3: State group attributes
    let n_state_attrs = 3;
    // 0: state_has_halt_transition
    // 1: state_has_self_loop
    // 2: state_is_halt_adjacent (can reach halt in 1 step)

    // Layer 4: Transition type attributes
    let trans_types = tm.transition_types();
    let n_trans_attrs = trans_types.len();

    let n_attrs = n_pos_attrs + n_global_attrs + n_state_attrs + n_trans_attrs;

    // Build transition attribute map
    let mut trans_to_attr: HashMap<(usize, usize), usize> = HashMap::new();
    for (ai, &(from, to)) in trans_types.iter().enumerate() {
        trans_to_attr.insert((from, to), n_pos_attrs + n_global_attrs + n_state_attrs + ai);
    }

    // Precompute state properties
    let mut state_has_halt: Vec<bool> = vec![false; n_states];
    let mut state_has_self_loop: Vec<bool> = vec![false; n_states];
    let mut state_halt_adjacent: Vec<bool> = vec![false; n_states];
    for si in 0..n_states {
        for sym in 0..2 {
            let t = &tm.transitions[si][sym];
            if t.next_state == n_states {
                state_has_halt[si] = true;
            }
            if t.next_state == si {
                state_has_self_loop[si] = true;
            }
            if t.next_state < n_states {
                // Check if next state has a halt transition
                for sym2 in 0..2 {
                    if tm.transitions[t.next_state][sym2].next_state == n_states {
                        state_halt_adjacent[si] = true;
                    }
                }
            }
        }
    }

    // ─── Object & Attribute Labels ────────────────────────────────────────

    let mut obj_labels: Vec<String> = Vec::with_capacity(n_objects);
    for (state, window) in &sampled {
        let window_str: String = window.iter()
            .map(|&b| if b == 0 { '0' } else { '1' })
            .collect();
        obj_labels.push(format!("{}[{}]", state_names[*state], window_str));
    }

    let mut attr_labels: Vec<String> = Vec::with_capacity(n_attrs);
    attr_labels.push("pos_even_0".to_string());
    attr_labels.push("pos_even_1".to_string());
    attr_labels.push("pos_odd_0".to_string());
    attr_labels.push("pos_odd_1".to_string());
    attr_labels.push("has_1s".to_string());
    attr_labels.push("all_0s".to_string());
    attr_labels.push("ones_lt_25".to_string());
    attr_labels.push("ones_25_50".to_string());
    attr_labels.push("ones_50_75".to_string());
    attr_labels.push("ones_gt_75".to_string());
    attr_labels.push("head_ctr".to_string());
    attr_labels.push("head_edge".to_string());
    attr_labels.push("st_has_halt".to_string());
    attr_labels.push("st_self_loop".to_string());
    attr_labels.push("st_halt_adj".to_string());
    for &(from, to) in &trans_types {
        attr_labels.push(format!("tr_{}→{}", state_names[from], state_names[to]));
    }

    // ─── Build Incidence Matrix ───────────────────────────────────────────

    let mut matrix = vec![vec![false; n_attrs]; n_objects];

    for (obj_idx, (state, window)) in sampled.iter().enumerate() {
        // Layer 1: Binned position-symbol attributes
        for pi in 0..n_positions {
            let sym = window[pi] as usize;
            matrix[obj_idx][pos_bin_attr(pi, sym)] = true;
        }

        // Layer 2: Global shared attributes
        let n_ones: usize = window.iter().filter(|&&b| b == 1).count();
        let density = n_ones as f64 / n_positions as f64;

        if n_ones > 0 { matrix[obj_idx][n_pos_attrs + 0] = true; }
        if n_ones == 0 { matrix[obj_idx][n_pos_attrs + 1] = true; }
        if density < 0.25 { matrix[obj_idx][n_pos_attrs + 2] = true; }
        if density >= 0.25 && density < 0.50 { matrix[obj_idx][n_pos_attrs + 3] = true; }
        if density >= 0.50 && density < 0.75 { matrix[obj_idx][n_pos_attrs + 4] = true; }
        if density >= 0.75 { matrix[obj_idx][n_pos_attrs + 5] = true; }

        // head position: center of window is at index window_l
        matrix[obj_idx][n_pos_attrs + 6] = true; // head is always at center

        // Layer 3: State group attributes
        let g_off = n_pos_attrs + n_global_attrs;
        if state_has_halt[*state] { matrix[obj_idx][g_off + 0] = true; }
        if state_has_self_loop[*state] { matrix[obj_idx][g_off + 1] = true; }
        if state_halt_adjacent[*state] { matrix[obj_idx][g_off + 2] = true; }

        // Layer 4: Transition attributes
        for sym in 0..2 {
            let t = &tm.transitions[*state][sym];
            if t.next_state < n_states {
                if let Some(&attr_idx) = trans_to_attr.get(&(*state, t.next_state)) {
                    matrix[obj_idx][attr_idx] = true;
                }
            }
        }
    }

    // Count unique objects (for display)
    let mut seen: HashSet<String> = HashSet::new();
    for (state, window) in &sampled {
        let key = format!("{}:{:?}", state, window);
        seen.insert(key);
    }

    println!(
        "    trace: {} steps ({} halted), {} objects ({} unique), {} attrs",
        _steps, if halted { "yes" } else { "no" },
        n_objects, seen.len(), n_attrs
    );

    (matrix, obj_labels, attr_labels)
}

/// Build the formal context from the TM's transition graph only.
/// This is a minimal lattice that captures the state transition structure,
/// useful as a baseline comparison.
pub fn build_transition_graph_context(
    tm: &TuringMachine,
) -> (Vec<Vec<bool>>, Vec<String>, Vec<String>) {
    let n_states = tm.n_states;
    let state_names: Vec<char> = ('A'..='Z').take(n_states).collect();
    let trans_types = tm.transition_types();

    // Objects = states, duplicated by symbol to create a richer lattice
    let n_objects = n_states * 2; // one per (state, symbol)
    // Attributes = next-state indicators + self-loop indicator
    let n_attrs = n_states + 1;

    let mut matrix = vec![vec![false; n_attrs]; n_objects];
    let mut obj_labels = Vec::with_capacity(n_objects);
    let mut attr_labels = Vec::with_capacity(n_attrs);

    for si in 0..n_states {
        for sym in 0..2 {
            let obj_idx = si * 2 + sym;
            obj_labels.push(format!("{}{}", state_names[si], sym));
            let t = &tm.transitions[si][sym];
            if t.next_state < n_states {
                matrix[obj_idx][t.next_state] = true;
            }
            // Self-loop indicator
            if t.next_state == si {
                matrix[obj_idx][n_states] = true;
            }
        }
    }

    for si in 0..n_states {
        attr_labels.push(format!("→{}", state_names[si]));
    }
    attr_labels.push("self_loop".to_string());

    (matrix, obj_labels, attr_labels)
}

// ─── Lattice Building & N-Operator Application ───────────────────────────────

/// Build the FCA lattice from the formal context matrix.
pub fn build_lattice(
    matrix: &[Vec<bool>],
    max_concepts: usize,
    time_limit: f64,
) -> FcaLattice {
    fca::build_synthetic_lattice(matrix, max_concepts, time_limit)
}

/// Run the N-operator iteration on the lattice and extract key metrics.
pub fn run_lattice_analysis(
    lattice: &FcaLattice,
    params: &DynamicsParams,
) -> (Vec<Option<IterResult>>, LatticeStats) {
    let stats = pipeline::compute_lattice_stats(lattice);
    let results = pipeline::run_topological_iteration(lattice, &stats, params);
    (results, stats)
}

/// Extract the spectral radius of the top (root) concept.
/// The top concept is the one with the highest height (closest to root).
pub fn extract_top_rho(
    results: &[Option<IterResult>],
    stats: &LatticeStats,
) -> Option<f64> {
    let n = results.len();
    if n == 0 {
        return None;
    }

    // Find the concept with maximum height (root/top)
    let mut max_height = 0;
    let mut top_idx = 0;
    for i in 0..n {
        if stats.heights[i] > max_height {
            max_height = stats.heights[i];
            top_idx = i;
        }
    }

    results[top_idx].as_ref().map(|r| r.rho_spectral)
}

/// Extract all spectral radii sorted by concept height.
pub fn extract_all_rho_by_height(
    results: &[Option<IterResult>],
    stats: &LatticeStats,
) -> Vec<(usize, f64)> {
    let mut pairs: Vec<(usize, f64)> = results
        .iter()
        .enumerate()
        .filter_map(|(i, opt)| {
            opt.as_ref().map(|r| (stats.heights[i], r.rho_spectral))
        })
        .collect();
    pairs.sort_by_key(|(h, _)| std::cmp::Reverse(*h));
    pairs
}

/// Compute ρ_top for increasing window sizes L, using the TRACE-BASED
/// formal context construction. The TM is simulated for max_steps before
/// extracting the lattice at each window size.
///
/// Returns a sequence of (L, n_concepts, n_edges, ρ_top, build_time, iter_time).
pub fn compute_rho_sequence(
    tm: &TuringMachine,
    max_l: usize,
    max_steps: u64,
    sample_every: u64,
    max_concepts: usize,
    time_limit: f64,
    params: &DynamicsParams,
) -> Vec<(usize, usize, usize, f64, f64, f64)> {
    let mut sequence = Vec::new();

    for l in 1..=max_l {
        let t0 = std::time::Instant::now();
        let (matrix, _obj_labels, _attr_labels) =
            build_trace_formal_context(tm, max_steps, l, sample_every);
        let lattice = build_lattice(&matrix, max_concepts, time_limit);
        let build_time = t0.elapsed().as_secs_f64();

        if lattice.concepts.is_empty() {
            eprintln!("  L={}: empty lattice, skipping", l);
            continue;
        }

        let t1 = std::time::Instant::now();
        let (results, stats) = run_lattice_analysis(&lattice, params);
        let iter_time = t1.elapsed().as_secs_f64();

        let rho_top = extract_top_rho(&results, &stats).unwrap_or(f64::NAN);

        sequence.push((
            l,
            lattice.concepts.len(),
            lattice.edges.len(),
            rho_top,
            build_time,
            iter_time,
        ));

        println!(
            "  L={:2}: concepts={:4}, edges={:4}, ρ_top={:.6}, build={:.2}s, iter={:.2}s",
            l, lattice.concepts.len(), lattice.edges.len(),
            rho_top, build_time, iter_time
        );
    }

    sequence
}

/// Compute ρ_top for the transition graph lattice (no trace, minimum version).
/// This uses the TM's transition graph structure only, without any computation.
pub fn compute_transition_graph_rho(
    tm: &TuringMachine,
    max_concepts: usize,
    time_limit: f64,
    params: &DynamicsParams,
) -> Option<(usize, usize, f64)> {
    let (matrix, _obj_labels, _attr_labels) = build_transition_graph_context(tm);
    let lattice = build_lattice(&matrix, max_concepts, time_limit);
    if lattice.concepts.is_empty() {
        return None;
    }
    let (results, stats) = run_lattice_analysis(&lattice, params);
    let rho_top = extract_top_rho(&results, &stats)?;
    Some((lattice.concepts.len(), lattice.edges.len(), rho_top))
}

// ─── Predefined BB(6) Machines ───────────────────────────────────────────────

/// Antihydra: the most famous BB(6) cryptid, Collatz-like behavior.
pub fn antihydra() -> TuringMachine {
    TuringMachine::parse(
        "Antihydra",
        "1RB1RA_0LC1LE_1LD1LC_1LA0LB_1LF1RE_---0RA"
    ).unwrap()
}

/// BB(6) Current Champion: known halter, >2↑↑↑5 steps.
pub fn current_champion() -> TuringMachine {
    TuringMachine::parse(
        "CurrentChampion",
        "1RB1RA_1RC1RZ_1LD0RF_1RA0LE_0LD1RC_1RA0RE"
    ).unwrap()
}

/// BB(5) champion for validation: known halter, 47,176,870 steps.
pub fn bb5_champion() -> TuringMachine {
    TuringMachine::parse(
        "BB5Champion",
        "1RB1LC_1RC1RB_1RD0LE_1LA1LD_1RZ0LA"
    ).unwrap()
}

/// A simple known halter for testing: 2-state, halts in 6 steps.
pub fn simple_halter() -> TuringMachine {
    TuringMachine::parse(
        "SimpleHalter",
        "1RB1RZ_1LA1RB"
    ).unwrap()
}

/// A simple known non-halter for testing: 2-state, never halts.
pub fn simple_nonhalter() -> TuringMachine {
    TuringMachine::parse(
        "SimpleNonHalter",
        "1RB1RB_1LA1LA"
    ).unwrap()
}

/// 3-state guaranteed non-halter: period-3 cycle, always writes 1.
/// More complex than the 2-state period-2 simple non-halter.
pub fn periodic3_nonhalter() -> TuringMachine {
    TuringMachine::parse(
        "Periodic3NonHalter",
        "1RB1RA_1RC1RB_1RA1RC"
    ).unwrap()
}

/// BB(4) champion: known halter, halts in 107 steps.
/// Represents an intermediate-complexity halting TM.
pub fn bb4_champion() -> TuringMachine {
    TuringMachine::parse(
        "BB4Champion",
        "1RB1LB_1LA0RC_1RZ1LD_1RD0RA"
    ).unwrap()
}

/// 4-state translator-type non-halter: moves right forever.
/// Generates a richer tape pattern than simple period-2.
pub fn translator_nonhalter() -> TuringMachine {
    TuringMachine::parse(
        "TranslatorNH",
        "1RB1LA_1LA0RB_1RB1RC_1LA1RD"
    ).unwrap()
}

// ─── v3: Information State Lattice (B-24.3) ──────────────────────────────────

/// Compute the 10-dimensional information state vector for a TM at a given step.
///
/// Components (B-24.3.2):
/// 1. Symbol density: s₁ = fraction of 1s in window
/// 2. Boundary entropy: s₂ = -[s₁·ln(s₁) + (1-s₁)·ln(1-s₁)] / ln(2)
/// 3. Run-length mean: s₃ = mean length of consecutive same-symbol runs
/// 4. Head position: s₄ = head_position / (L+1), normalized to [0,1]
/// 5. State activity: s₅ = fraction of states visited in recent L steps
/// 6. Transition entropy: s₆ = entropy of state transition frequencies
/// 7-10. Higher-order symbol correlations (lags 1, 2, 3)
pub fn compute_info_state(
    window: &[u8],
    head_pos: isize,
    window_l: usize,
    state: usize,
    n_states: usize,
    state_history: &[usize],  // recent L states visited (most recent last)
    trans_history: &[(usize, usize)], // recent L state transitions
) -> Vec<f64> {
    let n = window.len();
    let d = 10;
    let mut s = vec![0.0; d];

    if n == 0 {
        return s;
    }

    // 1. Symbol density
    let n_ones = window.iter().filter(|&&b| b == 1).count();
    s[0] = n_ones as f64 / n as f64;

    // 2. Boundary entropy (binary entropy normalized)
    let p = s[0].clamp(1e-10, 1.0 - 1e-10);
    s[1] = -(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) / std::f64::consts::LN_2;

    // 3. Run-length mean
    let mut runs: Vec<usize> = Vec::new();
    if n > 0 {
        let mut run_len = 1usize;
        for i in 1..n {
            if window[i] == window[i - 1] {
                run_len += 1;
            } else {
                runs.push(run_len);
                run_len = 1;
            }
        }
        runs.push(run_len);
    }
    if !runs.is_empty() {
        s[2] = runs.iter().sum::<usize>() as f64 / runs.len() as f64 / n as f64;
    }

    // 4. Head position normalized to [0, 1]
    // head_pos relative to window center: window[window_l] = head position
    // head_pos = actual head position on tape, window center = window_l
    // We compute: relative position in window
    let rel_pos = window_l as f64; // head is always at center of window
    s[3] = (rel_pos + 1.0) / (2.0 * window_l as f64 + 1.0);

    // 5. State activity: fraction of states visited in recent L steps
    if n_states > 0 && !state_history.is_empty() {
        let mut visited = vec![false; n_states];
        for &st in state_history {
            if st < n_states {
                visited[st] = true;
            }
        }
        visited[state] = true;
        let n_visited = visited.iter().filter(|&&v| v).count();
        s[4] = n_visited as f64 / n_states as f64;
    } else {
        s[4] = 1.0 / n_states as f64;
    }

    // 6. Transition entropy
    if !trans_history.is_empty() {
        let mut trans_counts: HashMap<(usize, usize), usize> = HashMap::new();
        for &(from, to) in trans_history {
            *trans_counts.entry((from, to)).or_insert(0) += 1;
        }
        let total = trans_history.len() as f64;
        let mut entropy = 0.0;
        for &count in trans_counts.values() {
            let prob = count as f64 / total;
            if prob > 0.0 {
                entropy -= prob * prob.ln();
            }
        }
        // Normalize by max entropy log(n_states^2) = 2*log(n_states)
        let max_entropy = 2.0 * (n_states as f64).ln();
        if max_entropy > 0.0 {
            s[5] = entropy / max_entropy;
        }
    }

    // 7-9. Higher-order symbol correlations (lags 1, 2, 3)
    for lag in 1..=3 {
        if n > lag {
            let mut sum_xy = 0.0;
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_x2 = 0.0;
            let mut sum_y2 = 0.0;
            let m = n - lag;
            for i in 0..m {
                let x = window[i] as f64;
                let y = window[i + lag] as f64;
                sum_xy += x * y;
                sum_x += x;
                sum_y += y;
                sum_x2 += x * x;
                sum_y2 += y * y;
            }
            let mf = m as f64;
            let num = mf * sum_xy - sum_x * sum_y;
            let den_x = (mf * sum_x2 - sum_x * sum_x).sqrt();
            let den_y = (mf * sum_y2 - sum_y * sum_y).sqrt();
            let den = den_x * den_y;
            let corr = if den > 1e-10 {
                (num / den).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            s[6 + lag - 1] = (corr + 1.0) / 2.0; // normalize to [0, 1]
        }
    }

    // 10. (s[9]) Cross-correlation between head position and symbol pattern
    // Use the correlation between a centered weight function and the tape
    let center = window_l as f64;
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;
    for i in 0..n {
        let dist = (i as f64 - center).abs();
        let weight = (-dist / (window_l as f64 + 1.0)).exp();
        weighted_sum += weight * window[i] as f64;
        total_weight += weight;
    }
    s[9] = if total_weight > 0.0 { weighted_sum / total_weight } else { 0.0 };

    s
}

/// Apply binning to an information state vector.
/// B = 4 bins per dimension, range [0, 1] → bin ∈ {0, 1, 2, 3}.
pub fn bin_info_state(s: &[f64], b: usize) -> Vec<usize> {
    s.iter()
        .map(|&v| {
            let clamped = v.clamp(0.0, 1.0 - 1e-10);
            ((clamped * b as f64) as usize).min(b - 1)
        })
        .collect()
}

/// Track state history and transition history during simulation.
pub struct InfoStateTracker {
    pub state_history: Vec<usize>,
    pub trans_history: Vec<(usize, usize)>,
    pub window_l: usize,
}

impl InfoStateTracker {
    pub fn new(window_l: usize) -> Self {
        InfoStateTracker {
            state_history: Vec::with_capacity(window_l * 2),
            trans_history: Vec::with_capacity(window_l * 2),
            window_l,
        }
    }

    pub fn record(&mut self, state: usize, prev_state: Option<usize>) {
        if let Some(prev) = prev_state {
            self.trans_history.push((prev, state));
        }
        self.state_history.push(state);
        // Keep only recent window_l entries
        while self.state_history.len() > self.window_l {
            self.state_history.remove(0);
        }
        while self.trans_history.len() > self.window_l {
            self.trans_history.remove(0);
        }
    }
}

/// Build the v3 information state formal context (B-24.3.4).
///
/// Objects = time steps (sampled)
/// Attributes = binned information state dimensions: bin_{k,b}
/// Total attributes = d * B = 10 * 4 = 40
pub fn build_info_state_context(
    tm: &TuringMachine,
    max_steps: u64,
    window_l: usize,
    sample_every: u64,
    b: usize, // number of bins per dimension
) -> (Vec<Vec<bool>>, Vec<String>, Vec<String>, Vec<Vec<f64>>) {
    let d = 10; // fixed 10-dimensional info state
    let n_attrs = d * b;

    // Step 1: Simulate the TM
    let (configs, halted, _steps) = tm.simulate(max_steps, window_l);

    // Step 2: Build info state trajectory with tracker
    let mut tracker = InfoStateTracker::new(window_l.max(10));
    let mut info_states: Vec<Vec<f64>> = Vec::new();
    let mut prev_state: Option<usize> = None;

    for (idx, (state, head, window)) in configs.iter().enumerate() {
        tracker.record(*state, prev_state);
        prev_state = Some(*state);

        if idx as u64 % sample_every != 0 {
            continue;
        }

        let info_vec = compute_info_state(
            window,
            *head,
            window_l,
            *state,
            tm.n_states,
            &tracker.state_history,
            &tracker.trans_history,
        );
        info_states.push(info_vec);
    }

    let n_objects = info_states.len();

    // ─── Attribute Labels ──────────────────────────────────────────────────
    let dim_names = [
        "density", "entropy", "runlen", "headpos",
        "st_active", "trans_ent", "corr_lag1", "corr_lag2",
        "corr_lag3", "weighted",
    ];

    let mut attr_labels: Vec<String> = Vec::with_capacity(n_attrs);
    for k in 0..d {
        for bin in 0..b {
            attr_labels.push(format!("{}_{}", dim_names[k], bin));
        }
    }

    // ─── Object Labels ─────────────────────────────────────────────────────
    let mut obj_labels: Vec<String> = Vec::with_capacity(n_objects);
    for (idx, s) in info_states.iter().enumerate() {
        let bins = bin_info_state(s, b);
        let bin_str: String = bins.iter()
            .map(|&bi| (b'0' + bi as u8) as char)
            .collect();
        obj_labels.push(format!("t{}:{}", idx * sample_every as usize, bin_str));
    }

    // ─── Build Incidence Matrix ────────────────────────────────────────────
    let mut matrix = vec![vec![false; n_attrs]; n_objects];

    for (obj_idx, s) in info_states.iter().enumerate() {
        let bins = bin_info_state(s, b);
        for (dim, &bin) in bins.iter().enumerate() {
            let attr_idx = dim * b + bin;
            matrix[obj_idx][attr_idx] = true;
        }
    }

    // Count unique bin vectors
    let mut seen_bins: HashSet<Vec<usize>> = HashSet::new();
    for s in &info_states {
        seen_bins.insert(bin_info_state(s, b));
    }

    println!(
        "    info_state: {} steps ({} halted), {} objects, {} unique bin vectors, {} attrs (d={} B={})",
        _steps, if halted { "yes" } else { "no" },
        n_objects, seen_bins.len(), n_attrs, d, b
    );

    (matrix, obj_labels, attr_labels, info_states)
}

/// Build the v4 information state formal context with **overlapping threshold attributes**.
///
/// Unlike v3 (mutually exclusive bins), this uses nested threshold attributes:
/// For each dimension k, create attributes `dim_k_gt_t1`, `dim_k_gt_t2`, ... where
/// t_j = j/(T+1) are evenly spaced thresholds. These are NOT mutually exclusive —
/// if value > t_j, then all > t_i for i<j are also true, creating rich overlapping patterns.
///
/// Total attributes = d * T (e.g., d=10, T=3 → 30 attrs)
pub fn build_info_state_context_overlap(
    tm: &TuringMachine,
    max_steps: u64,
    window_l: usize,
    sample_every: u64,
    n_thresholds: usize,
) -> (Vec<Vec<bool>>, Vec<String>, Vec<String>, Vec<Vec<f64>>) {
    let d = 10;
    let n_attrs = d * n_thresholds;

    let thresholds: Vec<f64> = (1..=n_thresholds)
        .map(|i| i as f64 / (n_thresholds + 1) as f64)
        .collect();

    let (configs, halted, _steps) = tm.simulate(max_steps, window_l);
    let mut tracker = InfoStateTracker::new(window_l.max(10));
    let mut info_states: Vec<Vec<f64>> = Vec::new();
    let mut prev_state: Option<usize> = None;

    for (idx, (state, head, window)) in configs.iter().enumerate() {
        tracker.record(*state, prev_state);
        prev_state = Some(*state);
        if idx as u64 % sample_every != 0 {
            continue;
        }
        let info_vec = compute_info_state(
            window, *head, window_l, *state, tm.n_states,
            &tracker.state_history, &tracker.trans_history,
        );
        info_states.push(info_vec);
    }

    let n_objects = info_states.len();

    let dim_names = [
        "density", "entropy", "runlen", "headpos",
        "st_active", "trans_ent", "corr_lag1", "corr_lag2",
        "corr_lag3", "weighted",
    ];

    let mut attr_labels: Vec<String> = Vec::with_capacity(n_attrs);
    for k in 0..d {
        for t in 0..n_thresholds {
            attr_labels.push(format!("{}_gt_{:.2}", dim_names[k], thresholds[t]));
        }
    }

    let mut obj_labels: Vec<String> = Vec::with_capacity(n_objects);
    for (idx, s) in info_states.iter().enumerate() {
        let mut n_true = 0usize;
        for k in 0..d {
            for t in 0..n_thresholds {
                if s[k] > thresholds[t] {
                    n_true += 1;
                }
            }
        }
        obj_labels.push(format!("t{}:{}attr", idx * sample_every as usize, n_true));
    }

    let mut matrix = vec![vec![false; n_attrs]; n_objects];
    for (obj_idx, s) in info_states.iter().enumerate() {
        for k in 0..d {
            for t in 0..n_thresholds {
                if s[k] > thresholds[t] {
                    matrix[obj_idx][k * n_thresholds + t] = true;
                }
            }
        }
    }

    let mut seen: HashSet<Vec<bool>> = HashSet::new();
    for row in &matrix {
        seen.insert(row.clone());
    }

    let mut attr_counts = vec![0usize; n_attrs];
    for row in &matrix {
        for (ai, &v) in row.iter().enumerate() {
            if v { attr_counts[ai] += 1; }
        }
    }
    let avg_density = attr_counts.iter().sum::<usize>() as f64 / n_attrs as f64 / n_objects as f64;

    println!(
        "    info_state(overlap): {} steps ({}), {} objs, {} unique pats, {} attrs (d={} T={}), attr_density={:.3}",
        _steps, if halted { "halted" } else { "non-halt" },
        n_objects, seen.len(), n_attrs, d, n_thresholds, avg_density
    );

    (matrix, obj_labels, attr_labels, info_states)
}

/// Compute ρ_top for increasing L using the v4 overlapping threshold lattice.
pub fn compute_info_state_rho_sequence_overlap(
    tm: &TuringMachine,
    max_l: usize,
    max_steps: u64,
    sample_every: u64,
    n_thresholds: usize,
    max_concepts: usize,
    time_limit: f64,
    params: &DynamicsParams,
) -> Vec<(usize, usize, usize, f64, f64, f64, usize)> {
    let mut sequence = Vec::new();

    for l in [5, 10, 20, 50, 100].iter().filter(|&&x| x <= max_l) {
        let t0 = std::time::Instant::now();
        let (matrix, _obj_labels, _attr_labels, _info_states) =
            build_info_state_context_overlap(tm, max_steps, *l, sample_every, n_thresholds);
        let lattice = build_lattice(&matrix, max_concepts, time_limit);
        let build_time = t0.elapsed().as_secs_f64();

        if lattice.concepts.is_empty() {
            eprintln!("  L={}: empty lattice, skipping", l);
            continue;
        }

        let t1 = std::time::Instant::now();
        let (results, stats) = run_lattice_analysis(&lattice, params);
        let iter_time = t1.elapsed().as_secs_f64();
        let rho_top = extract_top_rho(&results, &stats).unwrap_or(f64::NAN);

        let mut seen: HashSet<Vec<bool>> = HashSet::new();
        for row in &matrix {
            seen.insert(row.clone());
        }

        sequence.push((*l, lattice.concepts.len(), lattice.edges.len(),
            rho_top, build_time, iter_time, seen.len()));

        println!(
            "  L={:3}: concepts={:5}, edges={:5}, uniq_pats={:5}, ρ_top={:.8}, build={:.2}s, iter={:.2}s",
            l, lattice.concepts.len(), lattice.edges.len(),
            seen.len(), rho_top, build_time, iter_time
        );
    }

    sequence
}

// ─── v5: Pattern Transition Matrix Spectral Radius ────────────────────────────

/// Build a Markov transition matrix between unique attribute patterns.
///
/// Strategy (bypasses FCA entirely):
/// 1. Compute info state vectors and threshold them into attribute patterns
/// 2. Map each unique pattern to a state index
/// 3. Build transition count matrix T[i][j] = count(i→j)
/// 4. Normalize to Markov matrix M (row-stochastic)
/// 5. Compute ρ(M) = spectral radius
///
/// This directly measures TM behavioral complexity without FCA compression.
pub fn compute_pattern_transition_rho(
    tm: &TuringMachine,
    max_steps: u64,
    window_l: usize,
    sample_every: u64,
    n_thresholds: usize,
) -> (f64, usize, usize, f64, nalgebra::DMatrix<f64>) {
    let d = 10;
    let thresholds: Vec<f64> = (1..=n_thresholds)
        .map(|i| i as f64 / (n_thresholds + 1) as f64)
        .collect();

    let (configs, _halted, _steps) = tm.simulate(max_steps, window_l);
    let mut tracker = InfoStateTracker::new(window_l.max(10));
    let mut prev_state: Option<usize> = None;
    let mut patterns: Vec<Vec<bool>> = Vec::new();

    for (idx, (state, head, window)) in configs.iter().enumerate() {
        tracker.record(*state, prev_state);
        prev_state = Some(*state);
        if idx as u64 % sample_every != 0 {
            continue;
        }
        let info_vec = compute_info_state(
            window, *head, window_l, *state, tm.n_states,
            &tracker.state_history, &tracker.trans_history,
        );
        let mut pat = vec![false; d * n_thresholds];
        for k in 0..d {
            for t in 0..n_thresholds {
                if info_vec[k] > thresholds[t] {
                    pat[k * n_thresholds + t] = true;
                }
            }
        }
        patterns.push(pat);
    }

    let mut pattern_to_idx: HashMap<Vec<bool>, usize> = HashMap::new();
    let mut idx_to_pattern: Vec<Vec<bool>> = Vec::new();
    let mut pattern_seq: Vec<usize> = Vec::with_capacity(patterns.len());

    for pat in &patterns {
        let idx = *pattern_to_idx.entry(pat.clone()).or_insert_with(|| {
            let i = idx_to_pattern.len();
            idx_to_pattern.push(pat.clone());
            i
        });
        pattern_seq.push(idx);
    }

    let n_states_pat = idx_to_pattern.len();

    let mut trans_counts = nalgebra::DMatrix::<f64>::zeros(n_states_pat, n_states_pat);
    for w in pattern_seq.windows(2) {
        let from = w[0];
        let to = w[1];
        trans_counts[(from, to)] += 1.0;
    }

    let mut markov = trans_counts.clone();
    for i in 0..n_states_pat {
        let row_sum = markov.row(i).sum();
        if row_sum > 0.0 {
            for j in 0..n_states_pat {
                markov[(i, j)] /= row_sum;
            }
        }
    }

    let eigen = markov.clone().complex_eigenvalues();
    let rho = eigen.iter()
        .map(|c| c.norm())
        .fold(0.0_f64, f64::max);

    let total_transitions: f64 = trans_counts.sum();
    let mut stat_dist = vec![0.0; n_states_pat];
    for i in 0..n_states_pat {
        stat_dist[i] = trans_counts.row(i).sum() / total_transitions;
    }
    let entropy: f64 = stat_dist.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();

    (rho, n_states_pat, patterns.len(), entropy, markov)
}

/// Compute pattern transition ρ for multiple L values.
pub fn compute_pattern_transition_rho_sequence(
    tm: &TuringMachine,
    max_l: usize,
    max_steps: u64,
    sample_every: u64,
    n_thresholds: usize,
) -> Vec<(usize, f64, usize, usize, f64, f64)> {
    let mut sequence = Vec::new();

    for l in [5, 10, 20, 50, 100].iter().filter(|&&x| x <= max_l) {
        let t0 = std::time::Instant::now();
        let (rho, n_pats, n_samples, entropy, _markov) =
            compute_pattern_transition_rho(tm, max_steps, *l, sample_every, n_thresholds);
        let elapsed = t0.elapsed().as_secs_f64();

        sequence.push((*l, rho, n_pats, n_samples, entropy, elapsed));

        println!(
            "  L={:3}: ρ={:.8}, n_pats={:4}, n_samples={:5}, entropy={:.4}, time={:.2}s",
            l, rho, n_pats, n_samples, entropy, elapsed
        );
    }

    sequence
}

// ─── v6: k-gram Formal Context (ISD-native approach) ──────────────────────────

/// Extract k-grams from a window and return them as a vector of bit patterns.
/// Each k-gram is encoded as a usize (0..2^k).
fn extract_kgrams(window: &[u8], k: usize) -> Vec<usize> {
    if window.len() < k {
        return Vec::new();
    }
    let mut grams = Vec::with_capacity(window.len() - k + 1);
    for i in 0..=window.len() - k {
        let mut val: usize = 0;
        for j in 0..k {
            val = (val << 1) | (window[i + j] as usize);
        }
        grams.push(val);
    }
    grams
}

/// Build the formal context using k-grams of the TM tape as attributes.
///
/// ISD-native approach: the TM tape is treated as "text", k-grams as "words".
/// Objects = time windows, Attributes = k-grams, Incidence = k-gram appears in window.
pub fn build_kgram_formal_context(
    tm: &TuringMachine,
    max_steps: u64,
    window_l: usize,
    sample_every: u64,
    k: usize,
) -> (Vec<Vec<bool>>, Vec<String>, Vec<String>, usize) {
    let n_kgrams = 1usize << k;

    let (configs, halted, steps) = tm.simulate(max_steps, window_l);

    let mut windows: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut seen_kgrams: HashSet<usize> = HashSet::new();

    for (idx, (_state, _head, window)) in configs.iter().enumerate() {
        if idx as u64 % sample_every != 0 {
            continue;
        }
        let grams = extract_kgrams(window, k);
        for &g in &grams {
            seen_kgrams.insert(g);
        }
        windows.push((idx, grams));
    }

    let n_objects = windows.len();
    let active_kgrams: Vec<usize> = {
        let mut v: Vec<usize> = seen_kgrams.into_iter().collect();
        v.sort();
        v
    };
    let n_attrs = active_kgrams.len();

    let mut gram_to_idx: HashMap<usize, usize> = HashMap::new();
    for (i, &g) in active_kgrams.iter().enumerate() {
        gram_to_idx.insert(g, i);
    }

    let mut attr_labels: Vec<String> = Vec::with_capacity(n_attrs);
    for &g in &active_kgrams {
        attr_labels.push(format!("{:0width$b}", g, width = k));
    }

    let mut obj_labels: Vec<String> = Vec::with_capacity(n_objects);
    for (idx, grams) in &windows {
        obj_labels.push(format!("t{}:{}g", idx, grams.len()));
    }

    let mut matrix = vec![vec![false; n_attrs]; n_objects];
    for (obj_idx, (_idx, grams)) in windows.iter().enumerate() {
        for &g in grams {
            if let Some(&attr_idx) = gram_to_idx.get(&g) {
                matrix[obj_idx][attr_idx] = true;
            }
        }
    }

    let mut unique_patterns: HashSet<Vec<bool>> = HashSet::new();
    for row in &matrix {
        unique_patterns.insert(row.clone());
    }

    println!(
        "    kgram(k={}): {} steps ({}), {} windows, {} attrs (of {} possible), {} unique patterns",
        k, steps, if halted { "halted" } else { "non-halt" },
        n_objects, n_attrs, n_kgrams, unique_patterns.len()
    );

    (matrix, obj_labels, attr_labels, unique_patterns.len())
}

/// Build formal context using MULTI-SCALE k-grams as attributes.
///
/// B-24.20: Instead of a single k, use k ∈ [k_min, k_max] simultaneously.
/// Each attribute is (k, kgram_value), encoded as attr_id = (k << 12) | kgram_value.
/// Objects = time windows. Zero free parameters — k range is a structural choice.
///
/// The hypothesis: multi-scale k-grams enrich the attribute space, increasing
/// lattice size beyond the 10-concept threshold where ρ(J) can deviate from 0.54891053.
pub fn build_multiscale_kgram_context(
    tm: &TuringMachine,
    max_steps: u64,
    window_l: usize,
    sample_every: u64,
    k_min: usize,
    k_max: usize,
) -> (Vec<Vec<bool>>, Vec<String>, Vec<String>, usize) {
    let (configs, halted, steps) = tm.simulate(max_steps, window_l);

    let mut windows: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
    let mut seen_attrs: HashSet<(usize, usize)> = HashSet::new();

    for (idx, (_state, _head, window)) in configs.iter().enumerate() {
        if idx as u64 % sample_every != 0 { continue; }
        let mut all_grams: Vec<(usize, usize)> = Vec::new();
        for k in k_min..=k_max {
            let grams = extract_kgrams(window, k);
            for g in grams {
                let a = (k, g);
                seen_attrs.insert(a);
                all_grams.push(a);
            }
        }
        windows.push((idx, all_grams));
    }

    let n_objects = windows.len();
    let mut sorted_attrs: Vec<(usize, usize)> = seen_attrs.into_iter().collect();
    sorted_attrs.sort_by_key(|&(k, v)| (k, v));

    let mut attr_to_idx: HashMap<(usize, usize), usize> = HashMap::new();
    let mut attr_labels: Vec<String> = Vec::new();
    for &(k, v) in &sorted_attrs {
        attr_to_idx.insert((k, v), attr_labels.len());
        attr_labels.push(format!("k{}_{:0width$b}", k, v, width = k));
    }
    let n_attrs = attr_labels.len();

    let mut obj_labels: Vec<String> = Vec::with_capacity(n_objects);
    for (idx, grams) in &windows {
        obj_labels.push(format!("t{}:{}kg", idx, grams.len()));
    }

    let mut matrix = vec![vec![false; n_attrs]; n_objects];
    for (obj_idx, (_idx, grams)) in windows.iter().enumerate() {
        for &(k, v) in grams {
            if let Some(&attr_idx) = attr_to_idx.get(&(k, v)) {
                matrix[obj_idx][attr_idx] = true;
            }
        }
    }

    let mut unique_patterns: HashSet<Vec<bool>> = HashSet::new();
    for row in &matrix { unique_patterns.insert(row.clone()); }

    let total_kgrams_possible: usize = (k_min..=k_max).map(|k| 1usize << k).sum();
    println!(
        "    multiscale(k{}-{}): {} steps ({}), {} windows, {} attrs (of {} possible), {} unique patterns",
        k_min, k_max, steps, if halted { "halted" } else { "non-halt" },
        n_objects, n_attrs, total_kgrams_possible, unique_patterns.len()
    );

    (matrix, obj_labels, attr_labels, unique_patterns.len())
}

/// Build a TRANSPOSED formal context: objects = unique k-gram patterns, attributes = time windows.
///
/// B-24.18: Transposed FCA (Patterns-as-Objects).
/// This is the dual of build_kgram_formal_context:
///   - Objects = unique k-gram patterns (encoded as strings like "01011")
///   - Attributes = sampled time windows
///   - Entry (i,j) = 1 if pattern_i appears in time window j
///
/// Uses standard binary FCA only — NO Pattern Structure similarity operator, NO free parameters.
/// The transposition avoids the k-gram co-occurrence correlation degeneracy of v6.

// ─── v25: Prefix Containment Tree ──────────────────────────────────────────

/// B-24.21 gaze framework: prefix containment tree signature for a TM.
///
/// The containment tree T_k is a fixed complete binary tree of depth k.
/// A k-gram "11010" activates all its prefixes: "1101", "110", "11", "1", ε.
/// This is pure ⊑ (containment) — no Galois closure, no NextClosure.
///
/// Returns for each sampled time step:
///   - activated nodes this step
///   - cumulative activated set (概念 vs 要素)
///   - withdrawn nodes (activated at t-1 but not at t)
///   - containment depths of activated nodes
pub fn build_prefix_tree_signature(
    tm: &TuringMachine,
    max_steps: u64,
    window_l: usize,
    sample_every: u64,
    k: usize,
) -> PrefixTreeSignature {
    let (configs, _halted, _steps) = tm.simulate(max_steps, window_l);
    let depth_count = k + 1; // depths 0..k

    // Precompute node index: node with binary prefix `bits` at depth d
    // Total nodes = 2^(k+1) - 1
    let total_nodes = (1usize << (k + 1)) - 1;
    // Map: (depth, prefix_value) -> node_index (0..total_nodes-1)
    // Root ε = index 0, then BFS order: depth 1: "0"=1, "1"=2, depth 2: "00"=3, "01"=4, etc.
    // A cleaner mapping: node(depth, value) = (1<<depth) - 1 + value

    // Extract k-grams and activate prefixes per window
    let mut per_step_activated: Vec<Vec<usize>> = Vec::new();
    let mut per_step_new: Vec<Vec<usize>> = Vec::new();
    let mut per_step_withdrawn: Vec<Vec<usize>> = Vec::new();
    let mut cumulative: HashSet<usize> = HashSet::new();
    let mut prev_activated: HashSet<usize> = HashSet::new();

    for (idx, (_state, _head, window)) in configs.iter().enumerate() {
        if idx as u64 % sample_every != 0 { continue; }
        let grams = extract_kgrams(window, k);

        // Activate all prefixes of these k-grams
         let mut step_nodes: HashSet<usize> = HashSet::new();
         for g in grams {
             let mut prefix = g;
             for d in (0..=k).rev() {
                 let node_idx = (1usize << d) - 1 + prefix;
                 step_nodes.insert(node_idx);
                 prefix >>= 1; // next shorter prefix
             }
         }

        let new_nodes: Vec<usize> = step_nodes.iter()
            .filter(|&n| !cumulative.contains(n))
            .copied()
            .collect();
        let withdrawn: Vec<usize> = prev_activated.iter()
            .filter(|&n| !step_nodes.contains(n))
            .copied()
            .collect();

        per_step_activated.push(step_nodes.iter().copied().collect());
        per_step_new.push(new_nodes.clone());
        per_step_withdrawn.push(withdrawn.clone());

        for &n in &new_nodes { cumulative.insert(n); }
        prev_activated = step_nodes;
    }

    // Depth distribution of cumulative activated set
    let mut depth_counts = vec![0usize; depth_count];
    let mut depth_possible = vec![0usize; depth_count];
    for d in 0..=k {
        depth_possible[d] = 1usize << d;
    }
    for &n in &cumulative {
        // for node n, find its depth
        let mut depth = 0usize;
        let mut base = 0usize;
        for d in 0..=k {
            let start = (1usize << d) - 1;
            let end = (1usize << (d + 1)) - 1;
            if n >= start && n < end {
                depth = d;
                break;
            }
        }
        depth_counts[depth] += 1;
    }

    // Compute signatures per step
    let mut sigs: Vec<StepSignature> = Vec::new();
    for i in 0..per_step_activated.len() {
        let n_activated = per_step_activated[i].len();
        let n_new = per_step_new[i].len();
        let n_withdrawn = per_step_withdrawn[i].len();
        let n_cumulative = cumulative_at_step(&per_step_activated, i);

        // avg depth
        let mut depth_sum = 0usize;
        for &n in &per_step_activated[i] {
            for d in 0..=k {
                let start = (1usize << d) - 1;
                let end = (1usize << (d + 1)) - 1;
                if n >= start && n < end {
                    depth_sum += d;
                    break;
                }
            }
        }
        let avg_depth = if n_activated > 0 {
            depth_sum as f64 / n_activated as f64
        } else { 0.0 };

        sigs.push(StepSignature {
            step_idx: i,
            n_activated,
            n_new,
            n_withdrawn,
            n_cumulative,
            avg_depth,
        });
    }

    PrefixTreeSignature {
        k,
        total_nodes,
        n_steps: sigs.len(),
        cumulative_nodes: cumulative.len(),
        depth_counts,
        depth_possible,
        per_step: sigs,
    }
}

fn cumulative_at_step(per_step: &[Vec<usize>], up_to: usize) -> usize {
    let mut seen: HashSet<usize> = HashSet::new();
    for i in 0..=up_to {
        for &n in &per_step[i] { seen.insert(n); }
    }
    seen.len()
}

/// Structure returned by build_prefix_tree_signature
#[derive(Debug, Clone)]
pub struct PrefixTreeSignature {
    pub k: usize,
    pub total_nodes: usize,
    pub n_steps: usize,
    pub cumulative_nodes: usize,
    pub depth_counts: Vec<usize>,
    pub depth_possible: Vec<usize>,
    pub per_step: Vec<StepSignature>,
}

#[derive(Debug, Clone)]
pub struct StepSignature {
    pub step_idx: usize,
    pub n_activated: usize,
    pub n_new: usize,
    pub n_withdrawn: usize,
    pub n_cumulative: usize,
    pub avg_depth: f64,
}

// ─── v26: Downward Gaze (↓) Decomposition ─────────────────────────────────────

/// Decomposition metrics for a single time window.
#[derive(Debug, Clone)]
pub struct DownwardStepMetrics {
    pub step_idx: usize,
    /// How many activated nodes are decomposable (depth < k)?
    pub n_decomposable: usize,
    /// ↓_full: both children activated
    pub down_full: usize,
    /// ↓_partial: exactly one child activated
    pub down_partial: usize,
    /// ↓_empty: neither child activated
    pub down_empty: usize,
    /// ↓_support_ratio = down_full / n_decomposable
    pub down_support: f64,
}

/// Full signature of downward gaze across all time windows.
#[derive(Debug, Clone)]
pub struct DownwardGazeSignature {
    pub k: usize,
    pub total_nodes: usize,
    pub n_steps: usize,
    pub per_step: Vec<DownwardStepMetrics>,
    /// Aggregate across all windows
    pub agg_full: usize,
    pub agg_partial: usize,
    pub agg_empty: usize,
    pub agg_support: f64,
    /// ↓ decomposed children activation rate by depth
    pub depth_down_support: Vec<f64>,  // index = depth 0..k-1 (depth k has no children)
}

/// Build downward gaze (↓) decomposition signature.
///
/// For each time window:
/// 1. Apply ↑ gaze: activate all k-gram prefixes → C(t) (same as v25)
/// 2. For each activated node n ∈ C(t) at depth d < k:
///    ↓(n) = {n_left, n_right} — the two child prefixes at depth d+1
/// 3. Check if children are also activated in C(t) → classify as full/partial/empty
///
/// Secret classification (秘密归类): children found in C(t) are automatically
/// labeled as "part of n" — the decomposition relation n ⋈ {child₀, child₁}.
pub fn build_downward_gaze_signature(
    tm: &TuringMachine,
    max_steps: u64,
    window_l: usize,
    sample_every: u64,
    k: usize,
) -> DownwardGazeSignature {
    let (configs, _halted, _steps) = tm.simulate(max_steps, window_l);

    let total_nodes = (1usize << (k + 1)) - 1;
    let mut per_step: Vec<DownwardStepMetrics> = Vec::new();
    let mut agg_full = 0usize;
    let mut agg_partial = 0usize;
    let mut agg_empty = 0usize;
    // depth_down: [depth][full, partial, empty]
    let mut depth_down: Vec<(usize, usize, usize)> = vec![(0, 0, 0); k]; // depth 0..k-1

    for (idx, (_state, _head, window)) in configs.iter().enumerate() {
        if idx as u64 % sample_every != 0 { continue; }
        let grams = extract_kgrams(window, k);

        // ↑ gaze: activate all prefixes
        let mut step_nodes: HashSet<usize> = HashSet::new();
        for g in grams {
            let mut prefix = g;
            for d in (0..=k).rev() {
                let node_idx = (1usize << d) - 1 + prefix;
                step_nodes.insert(node_idx);
                prefix >>= 1;
            }
        }

        // ↓ gaze: for each activated node at depth < k, check children
        let mut n_decomposable = 0usize;
        let mut down_full = 0usize;
        let mut down_partial = 0usize;
        let mut down_empty = 0usize;

        for &node in &step_nodes {
            // Determine depth of node
            let mut depth = k + 1; // sentinel
            for d in 0..=k {
                let start = (1usize << d) - 1;
                let end = (1usize << (d + 1)) - 1;
                if node >= start && node < end {
                    depth = d;
                    break;
                }
            }
            if depth >= k { continue; } // leaf nodes can't decompose further

            n_decomposable += 1;

            // Children at depth d+1: left = (1<<(d+1))-1 + 2*value, right = left+1
            let value_at_d = node - ((1usize << depth) - 1);
            let left = (1usize << (depth + 1)) - 1 + 2 * value_at_d;
            let right = left + 1;

            let left_present = step_nodes.contains(&left);
            let right_present = step_nodes.contains(&right);

            if left_present && right_present {
                down_full += 1;
                depth_down[depth].0 += 1;
            } else if left_present || right_present {
                down_partial += 1;
                depth_down[depth].1 += 1;
            } else {
                down_empty += 1;
                depth_down[depth].2 += 1;
            }
        }

        let support = if n_decomposable > 0 {
            down_full as f64 / n_decomposable as f64
        } else { 0.0 };

        agg_full += down_full;
        agg_partial += down_partial;
        agg_empty += down_empty;

        per_step.push(DownwardStepMetrics {
            step_idx: per_step.len(),
            n_decomposable,
            down_full,
            down_partial,
            down_empty,
            down_support: support,
        });
    }

    let total_decomp = agg_full + agg_partial + agg_empty;
    let agg_support = if total_decomp > 0 {
        agg_full as f64 / total_decomp as f64
    } else { 0.0 };

    let depth_down_support: Vec<f64> = depth_down.iter().map(|&(f, p, e)| {
        let t = f + p + e;
        if t > 0 { f as f64 / t as f64 } else { 0.0 }
    }).collect();

    DownwardGazeSignature {
        k,
        total_nodes,
        n_steps: per_step.len(),
        per_step,
        agg_full,
        agg_partial,
        agg_empty,
        agg_support,
        depth_down_support,
    }
}

pub fn build_transposed_formal_context(
    configs: &[(usize, isize, Vec<u8>)],
    use_first: usize,
    sample_every: u64,
    k: usize,
) -> (Vec<Vec<bool>>, Vec<String>, Vec<String>, usize) {
    // Step 1: Extract k-grams from each sampled time window
    // map from pattern (usize encoding) to set of time window indices where it appears
    let mut pattern_windows: std::collections::HashMap<usize, std::collections::HashSet<usize>> =
        std::collections::HashMap::new();
    let mut window_indices: Vec<(usize, u64)> = Vec::new(); // (array_idx, actual_step)

    for (time_idx, (_state, _head, window)) in configs.iter().enumerate() {
        if time_idx >= use_first {
            break;
        }
        if time_idx as u64 % sample_every != 0 {
            continue;
        }
        let win_idx = window_indices.len();
        window_indices.push((time_idx, time_idx as u64));

        let grams = extract_kgrams(window, k);
        for g in grams {
            pattern_windows.entry(g).or_insert_with(std::collections::HashSet::new).insert(win_idx);
        }
    }

    let n_windows = window_indices.len();
    if n_windows == 0 {
        return (vec![], vec![], vec![], 0);
    }

    // Step 2: Build object list (unique patterns) and attribute list (time windows)
    let mut sorted_patterns: Vec<usize> = pattern_windows.keys().copied().collect();
    sorted_patterns.sort();
    let n_patterns = sorted_patterns.len();

    if n_patterns == 0 {
        return (vec![], vec![], vec![], 0);
    }

    // Object labels: binary representation of the pattern
    let obj_labels: Vec<String> = sorted_patterns.iter()
        .map(|&p| format!("{:0width$b}", p, width = k))
        .collect();

    // Attribute labels: step numbers
    let attr_labels: Vec<String> = window_indices.iter()
        .map(|(_, step)| format!("t{}", step))
        .collect();

    // Step 3: Build the binary matrix (patterns × windows)
    let mut matrix: Vec<Vec<bool>> = vec![vec![false; n_windows]; n_patterns];
    for (i, &pat) in sorted_patterns.iter().enumerate() {
        if let Some(windows) = pattern_windows.get(&pat) {
            for &w in windows {
                matrix[i][w] = true;
            }
        }
    }

    (matrix, obj_labels, attr_labels, n_patterns)
}

// ─── v24: Fuzzy FCA — Hamming-tolerance k-gram matching ──────────────────────

/// Hamming distance between two k-bit patterns.
#[inline]
fn hamming(a: usize, b: usize) -> u32 {
    (a ^ b).count_ones()
}

/// Given a k-gram pattern and Hamming radius h, return all patterns
/// within Hamming distance ≤ h (including the original).
fn hamming_neighbors(pattern: usize, k: usize, h: u32) -> Vec<usize> {
    let max_val = 1usize << k;
    let mut neighbors = Vec::new();
    for candidate in 0..max_val {
        if hamming(pattern, candidate) <= h {
            neighbors.push(candidate);
        }
    }
    neighbors
}

/// Build formal context with Hamming-tolerance fuzzy k-gram matching.
///
/// B-24.21: Instead of exact k-gram matching, a window "fuzzily contains"
/// all k-grams within Hamming distance ≤ h of any k-gram it actually contains.
///
/// This implements "共同属于" (joint belonging): a window doesn't need the
/// exact k-gram, just something close enough. h=0 recovers the standard
/// binary FCA (known to degenerate). h≥1 introduces fuzzy tolerance.
///
/// Zero free parameter: h is a discrete structural choice (0,1,2,...).
pub fn build_fuzzy_kgram_context(
    tm: &TuringMachine,
    max_steps: u64,
    window_l: usize,
    sample_every: u64,
    k: usize,
    h: u32,
) -> (Vec<Vec<bool>>, Vec<String>, Vec<String>, usize) {
    let (configs, _halted, steps) = tm.simulate(max_steps, window_l);

    let n_attrs = 1usize << k;
    let mut obj_vecs: Vec<(usize, Vec<bool>)> = Vec::new();

    // Precompute hamming-neighbor masks for all 2^k patterns
    let mut neighbor_map: Vec<Vec<usize>> = vec![Vec::new(); n_attrs];
    let neighbor_count: usize = neighbor_map.iter().map(|v| v.len()).sum();
    if h == 0 {
        for p in 0..n_attrs { neighbor_map[p] = vec![p]; }
    } else {
        for p in 0..n_attrs { neighbor_map[p] = hamming_neighbors(p, k, h); }
    }

    let mut total_hits = 0usize;
    for (idx, (_state, _head, window)) in configs.iter().enumerate() {
        if idx as u64 % sample_every != 0 { continue; }
        let grams = extract_kgrams(window, k);
        let mut row = vec![false; n_attrs];
        for g in grams {
            for &neighbor in &neighbor_map[g] {
                row[neighbor] = true;
            }
        }
        total_hits += row.iter().filter(|&&b| b).count();
        obj_vecs.push((idx, row));
    }

    let n_objects = obj_vecs.len();
    let density = if n_objects > 0 { total_hits as f64 / (n_objects * n_attrs) as f64 } else { 0.0 };

    let obj_labels: Vec<String> = obj_vecs.iter()
        .map(|(idx, _)| format!("t{}", idx))
        .collect();

    let attr_labels: Vec<String> = (0..n_attrs)
        .map(|p| format!("k{}_{:0width$b}", k, p, width = k))
        .collect();

    let matrix: Vec<Vec<bool>> = obj_vecs.into_iter().map(|(_, r)| r).collect();

    let mut unique_patterns: HashSet<Vec<bool>> = HashSet::new();
    for row in &matrix { unique_patterns.insert(row.clone()); }

    println!(
        "    fuzzy(k={} h={}): {} steps, {} objs, {} attrs, density={:.3}, {} uniq rows, {} neighbors total",
        k, h, steps, n_objects, n_attrs, density, unique_patterns.len(),
        neighbor_map.iter().map(|v| v.len()).sum::<usize>()
    );

    (matrix, obj_labels, attr_labels, unique_patterns.len())
}

/// Compute ρ(J) for k-gram FCA lattice at multiple L and k values.
pub fn compute_kgram_rho_sequence(
    tm: &TuringMachine,
    max_l: usize,
    max_steps: u64,
    sample_every: u64,
    k_values: &[usize],
    max_concepts: usize,
    time_limit: f64,
    params: &DynamicsParams,
) -> Vec<(usize, usize, usize, usize, usize, f64, f64, f64)> {
    let mut sequence = Vec::new();

    for &k in k_values {
        for l in [5, 10, 20, 50, 100].iter().filter(|&&x| x <= max_l) {
            let t0 = std::time::Instant::now();
            let (matrix, _obj_labels, _attr_labels, n_pats) =
                build_kgram_formal_context(tm, max_steps, *l, sample_every, k);
            let lattice = build_lattice(&matrix, max_concepts, time_limit);
            let build_time = t0.elapsed().as_secs_f64();

            if lattice.concepts.is_empty() {
                eprintln!("  k={} L={}: empty lattice, skipping", k, l);
                continue;
            }

            let t1 = std::time::Instant::now();
            let (results, stats) = run_lattice_analysis(&lattice, params);
            let iter_time = t1.elapsed().as_secs_f64();
            let rho_top = extract_top_rho(&results, &stats).unwrap_or(f64::NAN);

            sequence.push((k, *l, lattice.concepts.len(), lattice.edges.len(),
                n_pats, rho_top, build_time, iter_time));

            println!(
                "  k={} L={:3}: concepts={:5}, edges={:5}, uniq_pats={:5}, ρ_top={:.8}, build={:.2}s, iter={:.2}s",
                k, l, lattice.concepts.len(), lattice.edges.len(),
                n_pats, rho_top, build_time, iter_time
            );
        }
    }

    sequence
}

// ─── v7: Recursive Clustering Containment Extraction ────────────────────────

/// Compute a k-gram frequency vector for a tape window.
fn compute_kgram_freq(window: &[u8], k: usize) -> Vec<f64> {
    let n_grams = 1usize << k;
    let mut counts = vec![0.0; n_grams];
    if window.len() < k {
        return counts;
    }
    let n = (window.len() - k + 1) as f64;
    for i in 0..=(window.len() - k) {
        let mut val: usize = 0;
        for j in 0..k {
            val = (val << 1) | (window[i + j] as usize);
        }
        counts[val] += 1.0;
    }
    for g in 0..n_grams {
        counts[g] /= n;
    }
    counts
}

/// Simple k-means-like clustering.
fn cluster_vectors(vectors: &[Vec<f64>], n_clusters: usize, max_iters: usize) -> (Vec<usize>, Vec<Vec<f64>>) {
    let n = vectors.len();
    let dim = vectors[0].len();
    if n <= n_clusters {
        let mut centroids = vectors.to_vec();
        centroids.truncate(n_clusters);
        for _ in centroids.len()..n_clusters {
            centroids.push(vec![0.0; dim]);
        }
        let assignments: Vec<usize> = (0..n).map(|i| i.min(n_clusters - 1)).collect();
        return (assignments, centroids);
    }

    let mut rng = rand::thread_rng();
    let mut centroids: Vec<Vec<f64>> = (0..n_clusters)
        .map(|_| vectors[rng.gen_range(0..n)].clone())
        .collect();
    let mut assignments = vec![0usize; n];

    for _iter in 0..max_iters {
        let mut changed = false;
        for i in 0..n {
            let mut best = 0usize;
            let mut best_dist = f64::MAX;
            for c in 0..n_clusters {
                let mut dist = 0.0;
                for d in 0..dim {
                    let diff = vectors[i][d] - centroids[c][d];
                    dist += diff * diff;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        let mut new_centroids = vec![vec![0.0; dim]; n_clusters];
        let mut counts = vec![0usize; n_clusters];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for d in 0..dim {
                new_centroids[c][d] += vectors[i][d];
            }
        }
        for c in 0..n_clusters {
            if counts[c] > 0 {
                for d in 0..dim {
                    new_centroids[c][d] /= counts[c] as f64;
                }
            }
        }
        centroids = new_centroids;
    }

    (assignments, centroids)
}

/// Build containment hierarchy by recursive clustering.
///
/// Level 0: Windows -> k-gram frequency vectors
/// Level 1: Cluster windows into N_states states
/// Level 2: Build state transition matrix, cluster states into N_phases phases
///
/// Containment: window ⊂ state (membership), state ⊂ phase (membership)
pub fn build_recursive_clustering_context(
    tm: &TuringMachine,
    max_steps: u64,
    window_l: usize,
    sample_every: u64,
    k: usize,
    n_states: usize,
    n_phases: usize,
) -> (Vec<Vec<bool>>, Vec<String>, Vec<String>, (usize, usize, usize)) {
    let (configs, halted, steps) = tm.simulate(max_steps, window_l);

    let mut freq_vectors: Vec<Vec<f64>> = Vec::new();
    let mut window_indices: Vec<usize> = Vec::new();
    for (idx, (_state, _head, window)) in configs.iter().enumerate() {
        if idx as u64 % sample_every != 0 {
            continue;
        }
        freq_vectors.push(compute_kgram_freq(window, k));
        window_indices.push(idx);
    }

    let n_objects = freq_vectors.len();
    if n_objects < 5 {
        println!("    cluster: too few objects ({}), skipping", n_objects);
        return (vec![], vec![], vec![], (0, 0, 0));
    }

    let actual_states = n_states.min(n_objects);
    let (state_assign, _state_centroids) = cluster_vectors(&freq_vectors, actual_states, 20);

    let mut trans_counts = vec![vec![0usize; actual_states]; actual_states];
    for w in state_assign.windows(2) {
        trans_counts[w[0]][w[1]] += 1;
    }
    let mut trans_probs = vec![vec![0.0; actual_states]; actual_states];
    for i in 0..actual_states {
        let row_sum: usize = trans_counts[i].iter().sum();
        if row_sum > 0 {
            for j in 0..actual_states {
                trans_probs[i][j] = trans_counts[i][j] as f64 / row_sum as f64;
            }
        }
    }

    let actual_phases = n_phases.min(actual_states);
    let (phase_assign, _phase_centroids) = if actual_phases < actual_states {
        cluster_vectors(&trans_probs, actual_phases, 10)
    } else {
        ((0..actual_states).collect(), trans_probs.clone())
    };

    let n_attrs = actual_states + actual_phases;

    let mut attr_labels: Vec<String> = Vec::with_capacity(n_attrs);
    for s in 0..actual_states {
        let size = state_assign.iter().filter(|&&a| a == s).count();
        attr_labels.push(format!("state_{}:{}w", s, size));
    }
    for p in 0..actual_phases {
        let size = phase_assign.iter().filter(|&&a| a == p).count();
        attr_labels.push(format!("phase_{}:{}s", p, size));
    }

    let mut obj_labels: Vec<String> = Vec::with_capacity(n_objects);
    for (i, &idx) in window_indices.iter().enumerate() {
        let s = state_assign[i];
        let p = phase_assign[s];
        obj_labels.push(format!("t{}:s{}p{}", idx, s, p));
    }

    let mut matrix = vec![vec![false; n_attrs]; n_objects];
    for i in 0..n_objects {
        let s = state_assign[i];
        let p = phase_assign[s];
        matrix[i][s] = true;
        matrix[i][actual_states + p] = true;
    }

    let mut unique_patterns: HashSet<Vec<bool>> = HashSet::new();
    for row in &matrix {
        unique_patterns.insert(row.clone());
    }

    println!(
        "    cluster(k={} L={}): {} steps ({}), {} windows -> {} states -> {} phases, {} attrs, {} unique pats",
        k, window_l, steps, if halted { "halted" } else { "non-halt" },
        n_objects, actual_states, actual_phases, n_attrs, unique_patterns.len()
    );

    (matrix, obj_labels, attr_labels, (actual_states, actual_phases, unique_patterns.len()))
}

/// Compute ρ(J) for recursive clustering context at multiple (L, k, S, P) combos.
pub fn compute_cluster_rho_sequence(
    tm: &TuringMachine,
    max_l: usize,
    max_steps: u64,
    sample_every: u64,
    k_values: &[usize],
    state_sizes: &[usize],
    phase_sizes: &[usize],
    max_concepts: usize,
    time_limit: f64,
    params: &DynamicsParams,
) -> Vec<(usize, usize, usize, usize, usize, usize, f64, f64, f64)> {
    let mut sequence = Vec::new();

    for &k in k_values {
        for &l in [5, 10, 20, 50, 100].iter().filter(|&&x| x <= max_l) {
            for &ns in state_sizes {
                for &np in phase_sizes {
                    if np > ns {
                        continue;
                    }
                    let t0 = std::time::Instant::now();
                    let (matrix, _obj_labels, _attr_labels, (ns_act, np_act, _n_pats)) =
                        build_recursive_clustering_context(tm, max_steps, l, sample_every, k, ns, np);
                    if matrix.is_empty() {
                        continue;
                    }
                    let lattice = build_lattice(&matrix, max_concepts, time_limit);
                    let build_time = t0.elapsed().as_secs_f64();
                    if lattice.concepts.is_empty() {
                        eprintln!("  cluster k={} L={} S={} P={}: empty lattice", k, l, ns, np);
                        continue;
                    }
                    let t1 = std::time::Instant::now();
                    let (results, stats) = run_lattice_analysis(&lattice, params);
                    let iter_time = t1.elapsed().as_secs_f64();
                    let rho_top = extract_top_rho(&results, &stats).unwrap_or(f64::NAN);
                    sequence.push((k, l, ns_act, np_act,
                        lattice.concepts.len(), lattice.edges.len(),
                        rho_top, build_time, iter_time));
                    println!(
                        "  k={} L={:3} S={:2} P={:2}: concepts={:5}, edges={:5}, rho_top={:.8}, build={:.2}s, iter={:.2}s",
                        k, l, ns_act, np_act, lattice.concepts.len(),
                        lattice.edges.len(), rho_top, build_time, iter_time
                    );
                }
            }
        }
    }

    sequence
}

// ─── v8: Multi-level Block Decomposition (π mapping, B-24.5) ──────────────────

/// Extract non-overlapping blocks from a tape window at a given level.
///
/// B-24.5 π mapping: B_{r,i} = tape[i·2^r..(i+1)·2^r-1]
/// Level r: block size = 2^r, non-overlapping partition of the window.
fn extract_blocks(window: &[u8], level: usize) -> Vec<(usize, String)> {
    let block_size = 1usize << level;
    if window.len() < block_size {
        return Vec::new();
    }
    let n_blocks = window.len() / block_size;
    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let start = i * block_size;
        let end = start + block_size;
        let mut val: usize = 0;
        for j in start..end {
            val = (val << 1) | (window[j] as usize);
        }
        blocks.push((val, format!("L{}_B{}", level, i)));
    }
    blocks
}

/// Build formal context using multi-level block decomposition (v8 π mapping).
///
/// B-24.5: Non-overlapping power-of-2 blocks with natural containment.
/// Objects = all blocks across all levels and time steps.
/// Attributes = tape-position-based (pos_{k} for each window position k).
/// ALL levels share the same N_attr = window_l attribute space.
/// Block B_{r,i} has attribute pos_{k} iff tape[k] = 1 at the block's positions.
///
/// Key insight: blocks at DIFFERENT levels covering overlapping tape regions
/// SHARE attributes (containment). Blocks at SAME level cover DISJOINT regions
/// (no γ-correlation). This simultaneously satisfies containment and de-correlation.
pub fn build_block_decomposition_context(
    tm: &TuringMachine,
    max_steps: u64,
    window_l: usize,
    sample_every: u64,
    max_level: usize,
) -> (Vec<Vec<bool>>, Vec<String>, Vec<String>, usize, usize) {
    let (configs, halted, steps) = tm.simulate(max_steps, window_l);

    let n_attrs = window_l;
    let attr_labels: Vec<String> = (0..window_l).map(|k| format!("p{}", k)).collect();

    let mut all_block_data: Vec<(usize, usize, usize)> = Vec::new(); // (level, start_pos, time_idx)
    let mut block_labels: Vec<String> = Vec::new();
    let mut seen_patterns: HashSet<usize> = HashSet::new();

    for (time_idx, (_state, _head, window)) in configs.iter().enumerate() {
        if time_idx as u64 % sample_every != 0 {
            continue;
        }
        for level in 1..=max_level {
            let blocks = extract_blocks(window, level);
            for (content, label) in blocks {
                seen_patterns.insert(content);
                // start_pos = block_index * block_size
                let block_size = 1usize << level;
                let block_idx: usize = label
                    .strip_prefix(&format!("L{}_B", level))
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let start_pos = block_idx * block_size;
                all_block_data.push((level, start_pos, time_idx));
                block_labels.push(format!("t{}:{}", time_idx, label));
            }
        }
    }

    let n_objects = all_block_data.len();

    let mut matrix = vec![vec![false; n_attrs]; n_objects];

    // Fill incidence matrix: re-simulate using the same configs reference
    let mut obj_idx = 0usize;
    for (time_idx, (_state, _head, window)) in configs.iter().enumerate() {
        if time_idx as u64 % sample_every != 0 {
            continue;
        }
        for level in 1..=max_level {
            let block_size = 1usize << level;
            let n_blocks = window.len() / block_size;
            for block_i in 0..n_blocks {
                let start = block_i * block_size;
                for k in 0..block_size {
                    if window[start + k] == 1 {
                        // The tape position is start + k relative to window start
                        // Use a SHARED position attribute
                        let pos_attr = start + k;
                        if pos_attr < n_attrs {
                            matrix[obj_idx][pos_attr] = true;
                        }
                    }
                }
                obj_idx += 1;
            }
        }
    }

    let mut unique_row_patterns: HashSet<Vec<bool>> = HashSet::new();
    for row in &matrix {
        unique_row_patterns.insert(row.clone());
    }

    let mut level_counts = vec![0usize; max_level + 1];
    for &(level, _, _) in &all_block_data {
        level_counts[level] += 1;
    }

    let total_patterns = seen_patterns.len();

    println!(
        "    block_dec(L1-{}): {} steps ({}), {} objs, {} attrs, {} pats, {} uniq_rows",
        max_level, steps, if halted { "halted" } else { "non-halt" },
        n_objects, n_attrs, total_patterns, unique_row_patterns.len()
    );

    for level in 1..=max_level {
        if level_counts[level] > 0 {
            println!("      L{}: {} blocks", level, level_counts[level]);
        }
    }

    (matrix, block_labels, attr_labels, total_patterns, n_objects)
}

/// Compute ρ(J) for block decomposition FCA lattice at multiple (L, max_level) combos.
pub fn compute_block_rho_sequence(
    tm: &TuringMachine,
    max_l: usize,
    max_steps: u64,
    sample_every: u64,
    max_levels: &[usize],
    max_concepts: usize,
    time_limit: f64,
    params: &DynamicsParams,
) -> Vec<(usize, usize, usize, usize, usize, usize, f64, f64, f64)> {
    let mut sequence = Vec::new();

    for &ml in max_levels {
        for l in [5, 10, 20, 50, 100].iter().filter(|&&x| x <= max_l) {
            let t0 = std::time::Instant::now();
            let (matrix, _obj_labels, _attr_labels, n_pats, n_objs) =
                build_block_decomposition_context(tm, max_steps, *l, sample_every, ml);
            if matrix.is_empty() {
                continue;
            }
            let lattice = build_lattice(&matrix, max_concepts, time_limit);
            let build_time = t0.elapsed().as_secs_f64();

            if lattice.concepts.is_empty() {
                eprintln!("  block L={} ml={}: empty lattice, skipping", l, ml);
                continue;
            }

            let t1 = std::time::Instant::now();
            let (results, stats) = run_lattice_analysis(&lattice, params);
            let iter_time = t1.elapsed().as_secs_f64();
            let rho_top = extract_top_rho(&results, &stats).unwrap_or(f64::NAN);

            sequence.push((ml, *l, lattice.concepts.len(), lattice.edges.len(),
                n_pats, n_objs, rho_top, build_time, iter_time));

            println!(
                "  ml={} L={:3}: concepts={:5}, edges={:5}, pats={:5}, objs={:5}, ρ_top={:.8}, build={:.2}s, iter={:.2}s",
                ml, l, lattice.concepts.len(), lattice.edges.len(),
                n_pats, n_objs, rho_top, build_time, iter_time
            );
        }
    }

    sequence
}

// ─── v9: Pattern Structure π-Mapping (B-24.6) ─────────────────────────────────

/// Extract k-gram patterns from a bitvector (block content).
///
/// B-24.6: δ_k(B) = {B[t..t+k-1] : t = 0..|B|-k}
/// Key encoding: (k << 16) | pattern
/// Returns set of unique (k, pattern) keys.
fn extract_block_kgrams(content: usize, block_level: usize, k_min: usize, k_max: usize) -> Vec<usize> {
    let block_size = 1usize << block_level;
    if block_size < k_min {
        return Vec::new();
    }
    let eff_k_max = k_max.min(block_size);
    let mut pats = Vec::with_capacity((block_size - k_min + 1) * (eff_k_max - k_min + 1));
    for k in k_min..=eff_k_max {
        for t in 0..=(block_size - k) {
            let pat = (content >> (block_size - t - k)) & ((1usize << k) - 1);
            let key = (k << 16) | pat;
            pats.push(key);
        }
    }
    pats.sort();
    pats.dedup();
    pats
}

/// Build formal context using pattern structure (v9 π mapping).
///
/// B-24.6: Each block is described by the set of k-grams it contains.
/// Attributes = all unique k-gram patterns across all blocks.
/// This naturally bypasses the "impossible triangle" because:
/// - Non-overlapping blocks CAN share k-grams (content similarity)
/// - Containment is preserved: sub-block k-grams ⊆ super-block k-grams
/// - Each unique block content → distinct pattern set → distinct concept
pub fn build_pattern_structure_context(
    tm: &TuringMachine,
    max_steps: u64,
    window_l: usize,
    sample_every: u64,
    max_level: usize,
    k_min: usize,
    k_max: usize,
) -> (Vec<Vec<bool>>, Vec<String>, Vec<String>, usize, usize, usize) {
    let (configs, halted, steps) = tm.simulate(max_steps, window_l);

    let mut all_blocks: Vec<(usize, usize, usize, usize)> = Vec::new(); // (level, content, time, block_idx)
    let mut block_labels: Vec<String> = Vec::new();
    let mut pattern_map: HashMap<usize, Vec<usize>> = HashMap::new(); // content -> kgrams
    let mut all_patterns: HashSet<usize> = HashSet::new();

    for (time_idx, (_state, _head, window)) in configs.iter().enumerate() {
        if time_idx as u64 % sample_every != 0 {
            continue;
        }
        for level in 1..=max_level {
            let blocks = extract_blocks(window, level);
            for (content, label) in &blocks {
                if !pattern_map.contains_key(content) {
                    let pats = extract_block_kgrams(*content, level, k_min, k_max);
                    for &p in &pats { all_patterns.insert(p); }
                    pattern_map.insert(*content, pats);
                }
                let block_idx: usize = label
                    .strip_prefix(&format!("L{}_B", level))
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                all_blocks.push((level, *content, time_idx, block_idx));
                block_labels.push(format!("t{}:{}", time_idx, label));
            }
        }
    }

    let n_objects = all_blocks.len();

    let mut sorted_pats: Vec<usize> = all_patterns.iter().copied().collect();
    sorted_pats.sort();
    let mut pat_to_idx: HashMap<usize, usize> = HashMap::new();
    let mut attr_labels: Vec<String> = Vec::new();
    for &p in &sorted_pats {
        pat_to_idx.insert(p, attr_labels.len());
        let k = p >> 16;
        let val = p & 0xFFFF;
        attr_labels.push(format!("k{}_p{:0width$b}", k, val, width = k));
    }
    let n_attrs = attr_labels.len();

    // Build incidence matrix
    let mut matrix = vec![vec![false; n_attrs]; n_objects];
    for (obj_idx, &(_level, content, _time, _block_idx)) in all_blocks.iter().enumerate() {
        if let Some(pats) = pattern_map.get(&content) {
            for &p in pats {
                if let Some(&attr_idx) = pat_to_idx.get(&p) {
                    matrix[obj_idx][attr_idx] = true;
                }
            }
        }
    }

    let mut unique_rows: HashSet<Vec<bool>> = HashSet::new();
    for row in &matrix { unique_rows.insert(row.clone()); }

    let n_pats = all_patterns.len();

    let mut level_counts = vec![0usize; max_level + 1];
    for &(level, _, _, _) in &all_blocks { level_counts[level] += 1; }

    println!(
        "    ps(L1-{} k{}-{}): {} steps ({}), {} objs, {} attrs, {} pats, {} uniq_rows",
        max_level, k_min, k_max, steps, if halted { "halted" } else { "non-halt" },
        n_objects, n_attrs, n_pats, unique_rows.len()
    );
    for level in 1..=max_level {
        if level_counts[level] > 0 {
            println!("      L{}: {} blocks", level, level_counts[level]);
        }
    }

    (matrix, block_labels, attr_labels, n_pats, n_objects, n_attrs)
}

/// Compute ρ(J) for pattern structure FCA lattice at multiple (L, max_level, k_max) combos.
pub fn compute_pattern_rho_sequence(
    tm: &TuringMachine,
    max_l: usize,
    max_steps: u64,
    sample_every: u64,
    max_levels: &[usize],
    k_configs: &[(usize, usize)], // (k_min, k_max) combos
    max_concepts: usize,
    time_limit: f64,
    params: &DynamicsParams,
) -> Vec<(usize, usize, usize, usize, usize, usize, usize, usize, usize, f64, f64, f64)> {
    let mut sequence = Vec::new();

    for &ml in max_levels {
        for &(k_min, k_max) in k_configs {
            for l in [5, 10, 20, 50].iter().filter(|&&x| x <= max_l) {
                let t0 = std::time::Instant::now();
                let (matrix, _obj_labels, _attr_labels, n_pats, n_objs, n_attrs) =
                    build_pattern_structure_context(tm, max_steps, *l, sample_every, ml, k_min, k_max);
                if matrix.is_empty() { continue; }
                let lattice = build_lattice(&matrix, max_concepts, time_limit);
                let build_time = t0.elapsed().as_secs_f64();

                if lattice.concepts.is_empty() {
                    eprintln!("  ps ml={} k={}-{} L={}: empty lattice", ml, k_min, k_max, l);
                    continue;
                }

                let t1 = std::time::Instant::now();
                let (results, stats) = run_lattice_analysis(&lattice, params);
                let iter_time = t1.elapsed().as_secs_f64();
                let rho_top = extract_top_rho(&results, &stats).unwrap_or(f64::NAN);

                sequence.push((ml, k_min, k_max, *l, lattice.concepts.len(),
                    lattice.edges.len(), n_pats, n_objs, n_attrs, rho_top, build_time, iter_time));

                println!(
                    "  ps ml={} k={}-{} L={:3}: concepts={:6}, edges={:6}, pats={:6}, objs={:7}, attrs={:6}, ρ_top={:.8}, build={:.2}s, iter={:.2}s",
                    ml, k_min, k_max, *l, lattice.concepts.len(), lattice.edges.len(),
                    n_pats, n_objs, n_attrs, rho_top, build_time, iter_time
                );
            }
        }
    }

    sequence
}

// ─── v10: Random Matrix Spectral Diagnostics (B-24.7) ─────────────────────────

/// Full spectral diagnostics on a lattice: collect all eigenvalues,
/// compute spacing statistics, compare with Wigner-Dyson / Poisson.
#[derive(Debug, Clone)]
pub struct SpectralDiagnostic {
    pub name: String,
    pub n_concepts: usize,
    pub n_edges: usize,
    pub n_eigenvalues: usize,
    pub n_real: usize,
    pub n_complex: usize,
    pub rho_max: f64,
    pub rho_mean: f64,
    pub rho_median: f64,
    pub rho_std: f64,
    /// Real eigenvalue magnitudes, sorted ascending
    pub eigenvalues: Vec<f64>,
    /// Nearest-neighbor spacing (NNS) between consecutive eigenvalues
    pub spacings: Vec<f64>,
    /// NNS mean, std, min, max
    pub spacing_mean: f64,
    pub spacing_std: f64,
    /// Ratio statistics r_n = min(s_n,s_{n+1})/max(s_n,s_{n+1})
    pub ratios: Vec<f64>,
    pub ratio_mean: f64,
    /// Wigner-Dyson GOE reference: mean ratio ≈ 0.5359
    /// Poisson reference: mean ratio ≈ 0.386
    /// KS distance to GOE
    pub ks_goe: f64,
    pub ks_poisson: f64,
}

impl SpectralDiagnostic {
    /// GOE Wigner surmise CDF: F(s) = 1 - exp(-π s²/4)
    fn goe_cdf(x: f64) -> f64 { 1.0 - (-std::f64::consts::PI * x * x / 4.0).exp() }
    /// Poisson CDF: F(s) = 1 - exp(-s)
    fn poisson_cdf(x: f64) -> f64 { 1.0 - (-x).exp() }

    /// KS distance = max |empirical_cdf - theoretical_cdf|
    fn ks_distance(sorted: &[f64], cdf: fn(f64) -> f64) -> f64 {
        let n = sorted.len() as f64;
        sorted.iter().enumerate().fold(0.0, |max_d, (i, &x)| {
            let ecdf = (i + 1) as f64 / n;
            let d = (ecdf - cdf(x)).abs();
            if d > max_d { d } else { max_d }
        })
    }
}

pub fn compute_spectral_diagnostics(
    lattice: &FcaLattice,
    name: &str,
    params: &DynamicsParams,
) -> SpectralDiagnostic {
    let n_concepts = lattice.concepts.len();
    if n_concepts == 0 {
        return SpectralDiagnostic {
            name: name.to_string(), n_concepts: 0, n_edges: 0,
            n_eigenvalues: 0, n_real: 0, n_complex: 0,
            rho_max: f64::NAN, rho_mean: f64::NAN, rho_median: f64::NAN, rho_std: f64::NAN,
            eigenvalues: Vec::new(), spacings: Vec::new(),
            spacing_mean: f64::NAN, spacing_std: f64::NAN,
            ratios: Vec::new(), ratio_mean: f64::NAN,
            ks_goe: f64::NAN, ks_poisson: f64::NAN,
        };
    }

    let stats = pipeline::compute_lattice_stats(lattice);
    let results = pipeline::run_topological_iteration(lattice, &stats, params);

    // Collect ALL eigenvalue magnitudes from ALL concepts
    let mut all_eigs: Vec<f64> = Vec::new();
    let mut n_real = 0usize;
    let mut n_complex = 0usize;

    for opt in &results {
        if let Some(r) = opt {
            let j_arr = r.jacobian;
            let mut j = nalgebra::Matrix5::zeros();
            for row in 0..5 {
                for col in 0..5 {
                    j[(row, col)] = j_arr[row * 5 + col];
                }
            }
            let eigs = j.complex_eigenvalues();
            for c in &eigs {
                let mag = c.norm();
                if mag > 1e-15 {
                    all_eigs.push(mag);
                }
                if c.im.abs() < 1e-12 { n_real += 1; } else { n_complex += 1; }
            }
        }
    }

    let n_eigenvalues = all_eigs.len();
    all_eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let rho_max = all_eigs.last().copied().unwrap_or(f64::NAN);
    let rho_mean = if n_eigenvalues > 0 { all_eigs.iter().sum::<f64>() / n_eigenvalues as f64 } else { f64::NAN };
    let rho_median = if n_eigenvalues > 0 { all_eigs[n_eigenvalues / 2] } else { f64::NAN };
    let rho_std = if n_eigenvalues > 1 {
        let mean = rho_mean;
        (all_eigs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_eigenvalues as f64).sqrt()
    } else { f64::NAN };

    // Nearest-neighbor spacings (unfolded: divide by local mean spacing)
    let mut spacings: Vec<f64> = Vec::new();
    if n_eigenvalues >= 2 {
        let raw_spacings: Vec<f64> = all_eigs.windows(2).map(|w| w[1] - w[0]).collect();
        let raw_mean = raw_spacings.iter().sum::<f64>() / raw_spacings.len() as f64;
        spacings = raw_spacings.iter().map(|s| s / raw_mean).collect();
    }

    let spacing_mean = if !spacings.is_empty() { spacings.iter().sum::<f64>() / spacings.len() as f64 } else { f64::NAN };
    let spacing_std = if spacings.len() > 1 {
        let m = spacing_mean;
        (spacings.iter().map(|x| (x - m).powi(2)).sum::<f64>() / spacings.len() as f64).sqrt()
    } else { f64::NAN };

    // Adjacent eigenvalue ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})
    let mut ratios: Vec<f64> = Vec::new();
    if spacings.len() >= 2 {
        for w in spacings.windows(2) {
            let (a, b) = if w[0] <= w[1] { (w[0], w[1]) } else { (w[1], w[0]) };
            if b > 1e-15 { ratios.push(a / b); }
        }
    }
    let ratio_mean = if !ratios.is_empty() { ratios.iter().sum::<f64>() / ratios.len() as f64 } else { f64::NAN };

    // KS distance to GOE / Poisson
    let mut spacings_sorted = spacings.clone();
    spacings_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let ks_goe = if spacings_sorted.len() >= 5 {
        SpectralDiagnostic::ks_distance(&spacings_sorted, SpectralDiagnostic::goe_cdf)
    } else { f64::NAN };
    let ks_poisson = if spacings_sorted.len() >= 5 {
        SpectralDiagnostic::ks_distance(&spacings_sorted, SpectralDiagnostic::poisson_cdf)
    } else { f64::NAN };

    SpectralDiagnostic {
        name: name.to_string(),
        n_concepts,
        n_edges: lattice.edges.len(),
        n_eigenvalues,
        n_real,
        n_complex,
        rho_max, rho_mean, rho_median, rho_std,
        eigenvalues: all_eigs,
        spacings: spacings_sorted,
        spacing_mean, spacing_std,
        ratios,
        ratio_mean,
        ks_goe, ks_poisson,
    }
}

/// Run spectral diagnostics on pattern-structure FCA lattice for all 3 BB(6) TMs.
pub fn run_full_spectral_diagnostics(
    nonhalter: &TuringMachine,
    champion: &TuringMachine,
    antihydra: &TuringMachine,
    max_steps: u64,
    window_l: usize,
    sample_every: u64,
    max_level: usize,
    k_min: usize,
    k_max: usize,
    max_concepts: usize,
    time_limit: f64,
    params: &DynamicsParams,
) -> (SpectralDiagnostic, SpectralDiagnostic, SpectralDiagnostic) {
    println!("  Building pattern-structure lattices...\n");

    let (m_nh, _, _, _, _, _) = build_pattern_structure_context(nonhalter, max_steps, window_l, sample_every, max_level, k_min, k_max);
    let lattice_nh = build_lattice(&m_nh, max_concepts, time_limit);
    println!("    SimpleNonHalter: {} concepts, {} edges", lattice_nh.concepts.len(), lattice_nh.edges.len());

    let (m_cc, _, _, _, _, _) = build_pattern_structure_context(champion, max_steps, window_l, sample_every, max_level, k_min, k_max);
    let lattice_cc = build_lattice(&m_cc, max_concepts, time_limit);
    println!("    CurrentChampion: {} concepts, {} edges", lattice_cc.concepts.len(), lattice_cc.edges.len());

    let (m_ah, _, _, _, _, _) = build_pattern_structure_context(antihydra, max_steps, window_l, sample_every, max_level, k_min, k_max);
    let lattice_ah = build_lattice(&m_ah, max_concepts, time_limit);
    println!("    Antihydra:       {} concepts, {} edges", lattice_ah.concepts.len(), lattice_ah.edges.len());

    println!("\n  Computing full spectral diagnostics...\n");
    let diag_nh = compute_spectral_diagnostics(&lattice_nh, "SimpleNonHalter", params);
    let diag_cc = compute_spectral_diagnostics(&lattice_cc, "CurrentChampion", params);
    let diag_ah = compute_spectral_diagnostics(&lattice_ah, "Antihydra", params);

    (diag_nh, diag_cc, diag_ah)
}

// ─── Certificate Construction (per B-24 §4) ─────────────────────────────────

/// ISD Certificate for a Turing machine (B-24 §4.1).
#[derive(Debug, Clone)]
pub struct IsdCertificate {
    pub machine_name: String,
    pub std_format: String,
    /// ρ_top(L) sequence
    pub rho_sequence: Vec<(usize, f64)>,
    /// Window size at which numerical stability is reached
    pub l0: usize,
    /// Verdict: true = halting, false = non-halting
    pub verdict_halting: bool,
    /// Threshold for stability check
    pub epsilon: f64,
}

impl IsdCertificate {
    /// Construct certificate from ρ sequence.
    /// Uses criterion: ρ_top(L) stabilizes to 0 → halting; stabilizes to c > 0 → non-halting.
    pub fn from_sequence(
        machine_name: &str,
        std_format: &str,
        rho_sequence: &[(usize, f64)],
        epsilon: f64,
    ) -> Self {
        // Find L0: the first L where subsequent values are within epsilon
        let mut l0 = rho_sequence.last().map(|(l, _)| *l).unwrap_or(1);
        for i in 0..rho_sequence.len().saturating_sub(1) {
            let mut stable = true;
            for j in (i + 1)..rho_sequence.len() {
                if (rho_sequence[j].1 - rho_sequence[i].1).abs() > epsilon {
                    stable = false;
                    break;
                }
            }
            if stable {
                l0 = rho_sequence[i].0;
                break;
            }
        }

        // Final ρ value
        let final_rho = rho_sequence.last().map(|(_, r)| *r).unwrap_or(1.0);

        // Verdict: ρ < epsilon → halting, ρ > epsilon → non-halting
        // (Using the threshold from B-24 §3.2: ε = 10^{-3})
        let verdict_halting = final_rho < epsilon;

        IsdCertificate {
            machine_name: machine_name.to_string(),
            std_format: std_format.to_string(),
            rho_sequence: rho_sequence.iter().map(|(l, r)| (*l, *r)).collect(),
            l0,
            verdict_halting,
            epsilon,
        }
    }

    /// Verify the certificate (B-24 §4.2).
    pub fn verify(&self) -> bool {
        // Check 1: All ρ values are in (0, 1)
        for &(_l, rho) in &self.rho_sequence {
            if rho <= 0.0 || rho >= 1.0 || rho.is_nan() {
                return false;
            }
        }

        // Check 2: Sequence stabilizes after L0
        let l0_rho = self.rho_sequence
            .iter()
            .find(|(l, _)| *l == self.l0)
            .map(|(_, r)| *r);

        if let Some(l0_rho) = l0_rho {
            for &(l, rho) in &self.rho_sequence {
                if l >= self.l0 && (rho - l0_rho).abs() > self.epsilon {
                    return false;
                }
            }
        }

        // Check 3: Verdict is consistent with the criterion
        let final_rho = self.rho_sequence.last().map(|(_, r)| *r).unwrap_or(1.0);
        let expected_verdict = final_rho < self.epsilon;
        if self.verdict_halting != expected_verdict {
            return false;
        }

        true
    }

    pub fn display(&self) {
        println!("\n┌─────────────────────────────────────────────────┐");
        println!("│  ISD Certificate: {} │", self.machine_name);
        println!("├─────────────────────────────────────────────────┤");
        println!("│  Format: {} │", self.std_format);
        println!("│  L₀ (stability): {}                               │", self.l0);
        println!("│  ε (threshold):  {}                              │", self.epsilon);
        println!("│  Verdict: {}                    │",
            if self.verdict_halting { "HALTING    " } else { "NON-HALTING" });
        println!("├─────────────────────────────────────────────────┤");
        println!("│  ρ_top(L) sequence:                              │");
        for &(l, rho) in &self.rho_sequence {
            let marker = if l >= self.l0 { " ✓" } else { "" };
            println!("│    L={:2}: ρ={:.8}{}                         │", l, rho, marker);
        }
        println!("├─────────────────────────────────────────────────┤");
        println!("│  Verified: {}                                  │",
            if self.verify() { "PASS" } else { "FAIL" });
        println!("└─────────────────────────────────────────────────┘");
    }
}

// ─── v11: Kolmogorov-Sinai Entropy (Dynamical Systems, B-24.8) ───────────────

/// Compute block entropy H(L) for binary tape sequences.
/// H(L) = -Σ p(s) log₂ p(s) over all length-L blocks occurring in the tape.
/// Returns H(1), H(2), …, H(max_len) and the entropy rate estimate.
pub fn compute_ks_entropy(
    tm: &TuringMachine,
    max_steps: u64,
    window_l: usize,
    max_block_len: usize,
) -> (Vec<f64>, f64, u64) {
    let (configs, _halted, steps) = tm.simulate(max_steps, window_l);
    let tape_len = if let Some((_, _, w)) = configs.first() { w.len() } else { return (vec![], 0.0, 0); };
    let n_samples = configs.len().min(1000);

    let mut h_vals = Vec::with_capacity(max_block_len);

    for len in 1..=max_block_len {
        if len > tape_len { break; }
        let mut counts: HashMap<usize, usize> = HashMap::new();
        let mut total = 0usize;

        for (ti, (_, _, window)) in configs.iter().enumerate() {
            if ti >= n_samples { break; }
            for start in 0..=(window.len().saturating_sub(len)) {
                let mut val = 0usize;
                for k in 0..len {
                    val = (val << 1) | (window[start + k] as usize);
                }
                *counts.entry(val).or_insert(0) += 1;
                total += 1;
            }
        }

        if total == 0 { break; }

        let entropy: f64 = counts.values().map(|&c| {
            let p = c as f64 / total as f64;
            -p * p.log2()
        }).sum();

        h_vals.push(entropy);
    }

    // Entropy rate h = lim_{L→∞} H(L)/L, estimated as H(L) - H(L-1) for largest L
    let rate = if h_vals.len() >= 2 {
        let last = h_vals[h_vals.len() - 1];
        let prev = h_vals[h_vals.len() - 2];
        (last - prev).max(0.0)
    } else {
        h_vals.first().copied().unwrap_or(0.0)
    };

    (h_vals, rate, steps)
}

// ─── Time Evolution: Partial Trajectory Analysis ─────────────────────────────

pub fn compute_ks_entropy_from_configs(
    configs: &[(usize, isize, Vec<u8>)],
    use_first: usize,
    max_block_len: usize,
) -> (Vec<f64>, f64) {
    if configs.is_empty() {
        return (vec![], 0.0);
    }
    let tape_len = configs[0].2.len();
    let n_samples = use_first.min(configs.len());

    let mut h_vals = Vec::with_capacity(max_block_len);
    for len in 1..=max_block_len {
        if len > tape_len {
            break;
        }
        let mut counts: HashMap<usize, usize> = HashMap::new();
        let mut total = 0usize;
        for ti in 0..n_samples {
            let window = &configs[ti].2;
            for start in 0..=(window.len().saturating_sub(len)) {
                let mut val = 0usize;
                for k in 0..len {
                    val = (val << 1) | (window[start + k] as usize);
                }
                *counts.entry(val).or_insert(0) += 1;
                total += 1;
            }
        }
        if total == 0 {
            break;
        }
        let h: f64 = counts.values().map(|&c| {
            if c == 0 {
                0.0
            } else {
                let p = c as f64 / total as f64;
                -p * p.log2()
            }
        }).sum::<f64>() / len as f64;
        h_vals.push(h);
    }

    let rate = if h_vals.len() >= 2 {
        let last = h_vals[h_vals.len() - 1];
        let prev = h_vals[h_vals.len() - 2];
        (last - prev).max(0.0)
    } else {
        h_vals.first().copied().unwrap_or(0.0)
    };

    (h_vals, rate)
}

pub fn build_pattern_structure_context_from_configs(
    configs: &[(usize, isize, Vec<u8>)],
    use_first: usize,
    sample_every: u64,
    max_level: usize,
    k_min: usize,
    k_max: usize,
) -> (Vec<Vec<bool>>, Vec<String>, Vec<String>, usize, usize, usize) {
    let mut all_blocks: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut block_labels: Vec<String> = Vec::new();
    let mut pattern_map: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut all_patterns: HashSet<usize> = HashSet::new();

    for (time_idx, (_state, _head, window)) in configs.iter().enumerate() {
        if time_idx >= use_first {
            break;
        }
        if time_idx as u64 % sample_every != 0 {
            continue;
        }
        for level in 1..=max_level {
            let blocks = extract_blocks(window, level);
            for (content, label) in &blocks {
                if !pattern_map.contains_key(content) {
                    let pats = extract_block_kgrams(*content, level, k_min, k_max);
                    for &p in &pats {
                        all_patterns.insert(p);
                    }
                    pattern_map.insert(*content, pats);
                }
                let block_idx: usize = label
                    .strip_prefix(&format!("L{}_B", level))
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                all_blocks.push((level, *content, time_idx, block_idx));
                block_labels.push(format!("t{}:{}", time_idx, label));
            }
        }
    }

    let n_objects = all_blocks.len();
    let mut sorted_pats: Vec<usize> = all_patterns.iter().copied().collect();
    sorted_pats.sort();
    let mut pat_to_idx: HashMap<usize, usize> = HashMap::new();
    let mut attr_labels: Vec<String> = Vec::new();
    for &p in &sorted_pats {
        pat_to_idx.insert(p, attr_labels.len());
        let k = p >> 16;
        let val = p & 0xFFFF;
        attr_labels.push(format!("k{}_p{:0width$b}", k, val, width = k));
    }
    let n_attrs = attr_labels.len();

    if n_objects == 0 || n_attrs == 0 {
        return (vec![], vec![], vec![], 0, 0, 0);
    }

    let mut matrix: Vec<Vec<bool>> = vec![vec![false; n_attrs]; n_objects];
    for (obj_idx, (_, content, _, _)) in all_blocks.iter().enumerate() {
        if let Some(pats) = pattern_map.get(content) {
            for &p in pats {
                if let Some(&attr_idx) = pat_to_idx.get(&p) {
                    matrix[obj_idx][attr_idx] = true;
                }
            }
        }
    }

    (
        matrix,
        block_labels,
        attr_labels,
        sorted_pats.len(),
        n_objects,
        n_attrs,
    )
}

/// Track ISD complexity indicators across time for a single TM.
/// Returns vectors of (step, concepts, ks_rate, h1) for each checkpoint.
/// ks_rate = H(5)-H(4), h1 = raw H(1) entropy per bit.
pub fn compute_time_evolution(
    tm: &TuringMachine,
    max_steps: u64,
    checkpoints: &[u64],
    window_l: usize,
    max_level: usize,
    k_min: usize,
    k_max: usize,
    max_concepts: usize,
    time_limit: f64,
) -> Vec<(u64, usize, f64, f64)> {
    let (configs, _halted, _steps) = tm.simulate(max_steps, window_l);

    let mut results = Vec::new();

    for &cp in checkpoints {
        let use_n = (cp as usize).min(configs.len());

        let (h_vals, ks_rate) = compute_ks_entropy_from_configs(&configs, use_n, 5);
        let h1 = h_vals.first().copied().unwrap_or(0.0);

        let (matrix, _, _, _, _, _) = build_pattern_structure_context_from_configs(
            &configs, use_n, 4, max_level, k_min, k_max,
        );

        let n_concepts = if matrix.is_empty() || matrix[0].len() == 0 {
            0
        } else {
            let lattice = build_lattice(&matrix, max_concepts, time_limit);
            lattice.concepts.len()
        };

        results.push((cp, n_concepts, ks_rate, h1));
    }

    results
}

pub fn compute_lattice_depth_stats(lattice: &FcaLattice) -> (usize, Vec<usize>, f64) {
    let n = lattice.concepts.len();
    if n == 0 {
        return (0, vec![], 0.0);
    }

    let mut children: Vec<Vec<usize>> = vec![vec![]; n];
    let mut in_degree: Vec<usize> = vec![0; n];
    for &(parent, child) in &lattice.edges {
        if parent < n && child < n {
            children[parent].push(child);
            in_degree[child] += 1;
        }
    }

    let root = (0..n).min_by_key(|&i| in_degree[i]).unwrap_or(0);

    let mut depth = vec![0usize; n];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(root);
    depth[root] = 0;

    while let Some(u) = queue.pop_front() {
        for &v in &children[u] {
            if depth[v] == 0 && v != root {
                depth[v] = depth[u] + 1;
                queue.push_back(v);
            }
        }
    }

    let max_d = *depth.iter().max().unwrap_or(&0);
    let mut hist = vec![0usize; max_d + 1];
    for &d in &depth {
        hist[d] += 1;
    }
    let mean_d = depth.iter().sum::<usize>() as f64 / n as f64;

    (max_d, hist, mean_d)
}

pub fn compute_time_evolution_rich(
    tm: &TuringMachine,
    max_steps: u64,
    checkpoints: &[u64],
    window_l: usize,
    max_level: usize,
    k_min: usize,
    k_max: usize,
    max_concepts: usize,
    time_limit: f64,
) -> Vec<(u64, usize, f64, f64, usize, f64)> {
    let (configs, _halted, _steps) = tm.simulate(max_steps, window_l);

    let mut results = Vec::new();

    for &cp in checkpoints {
        let use_n = (cp as usize).min(configs.len());

        let (h_vals, ks_rate) = compute_ks_entropy_from_configs(&configs, use_n, 5);
        let h1 = h_vals.first().copied().unwrap_or(0.0);

        let (matrix, _, _, _, _, _) = build_pattern_structure_context_from_configs(
            &configs, use_n, 4, max_level, k_min, k_max,
        );

        let (n_concepts, max_depth, mean_depth) = if matrix.is_empty() || matrix[0].len() == 0 {
            (0, 0, 0.0)
        } else {
            let lattice = build_lattice(&matrix, max_concepts, time_limit);
            let nc = lattice.concepts.len();
            let (md, _, mean_d) = compute_lattice_depth_stats(&lattice);
            (nc, md, mean_d)
        };

        results.push((cp, n_concepts, ks_rate, h1, max_depth, mean_depth));
    }

    results
}

pub fn compute_time_evolution_with_turnover(
    tm: &TuringMachine,
    max_steps: u64,
    checkpoints: &[u64],
    window_l: usize,
    max_level: usize,
    k_min: usize,
    k_max: usize,
    max_concepts: usize,
    time_limit: f64,
) -> Vec<(u64, usize, f64, f64, usize, f64, f64, f64)> {
    let (configs, _halted, _steps) = tm.simulate(max_steps, window_l);

    let mut results: Vec<(u64, usize, f64, f64, usize, f64, Option<Vec<Vec<usize>>>)> = Vec::new();

    for &cp in checkpoints {
        let use_n = (cp as usize).min(configs.len());

        let (h_vals, ks_rate) = compute_ks_entropy_from_configs(&configs, use_n, 5);
        let h1 = h_vals.first().copied().unwrap_or(0.0);

        let (matrix, _, _, _, _, _) = build_pattern_structure_context_from_configs(
            &configs, use_n, 4, max_level, k_min, k_max,
        );

        let (n_concepts, max_depth, mean_depth, intents) = if matrix.is_empty() || matrix[0].len() == 0 {
            (0, 0, 0.0, None)
        } else {
            let lattice = build_lattice(&matrix, max_concepts, time_limit);
            let nc = lattice.concepts.len();
            let (md, _, mean_d) = compute_lattice_depth_stats(&lattice);
            let intents: Vec<Vec<usize>> = lattice.concepts.iter()
                .map(|c| {
                    let mut v = c.intent.clone();
                    v.sort();
                    v
                })
                .collect();
            (nc, md, mean_d, Some(intents))
        };

        results.push((cp, n_concepts, ks_rate, h1, max_depth, mean_depth, intents));
    }

    let mut output: Vec<(u64, usize, f64, f64, usize, f64, f64, f64)> = Vec::new();

    for i in 0..results.len() {
        let (step, nc, ks, h1, md, mean_d, ref _intents) = results[i];

        let jaccard = if i > 0 {
            if let (Some(prev), Some(curr)) = (&results[i - 1].6, &results[i].6) {
                let prev_set: HashSet<Vec<usize>> = prev.iter().cloned().collect();
                let curr_set: HashSet<Vec<usize>> = curr.iter().cloned().collect();
                let inter = prev_set.intersection(&curr_set).count();
                let union = prev_set.union(&curr_set).count();
                if union > 0 { inter as f64 / union as f64 } else { 1.0 }
            } else {
                0.0
            }
        } else {
            1.0
        };

        let (ncs, h1s): (Vec<f64>, Vec<f64>) = results.iter()
            .map(|(_, n, _, h, _, _, _)| (*n as f64, *h))
            .unzip();

        let xc = if i < results.len() - 1 {
            let n_now = nc as f64;
            let h_next = results[i + 1].3;
            let (n_mean, h_mean): (f64, f64) = (
                ncs.iter().sum::<f64>() / ncs.len() as f64,
                h1s.iter().sum::<f64>() / h1s.len() as f64,
            );
            let num = (n_now - n_mean) * (h_next - h_mean);
            let dn: f64 = ncs.iter().map(|x| (x - n_mean).powi(2)).sum::<f64>().sqrt();
            let dh: f64 = h1s.iter().map(|x| (x - h_mean).powi(2)).sum::<f64>().sqrt();
            if dn > 0.0 && dh > 0.0 {
                num / (dn * dh)
            } else {
                0.0
            }
        } else {
            0.0
        };

        output.push((step, nc, ks, h1, md, mean_d, jaccard, xc));
    }

    output
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_antihydra() {
        let tm = antihydra();
        assert_eq!(tm.n_states, 6);
        // Check a few transitions
        assert_eq!(tm.transitions[0][0], // A,0 -> 1RB
            Transition { next_state: 1, write: 1, direction: 1 });
        assert_eq!(tm.transitions[5][0], // F,0 -> --- (halt)
            Transition { next_state: 6, write: 0, direction: 0 });
    }

    #[test]
    fn test_parse_champion() {
        let tm = current_champion();
        assert_eq!(tm.n_states, 6);
        // B,1 -> 1RZ (halt)
        assert_eq!(tm.transitions[1][1],
            Transition { next_state: 6, write: 1, direction: 1 });
    }

    #[test]
    fn test_parse_bb5() {
        let tm = bb5_champion();
        assert_eq!(tm.n_states, 5);
    }

    #[test]
    fn test_formal_context_small() {
        let tm = simple_halter();
        let (matrix, obj_labels, attr_labels) = build_trace_formal_context(&tm, 100, 2, 1);
        assert!(!matrix.is_empty());
        assert_eq!(obj_labels.len(), matrix.len());
        assert_eq!(attr_labels.len(), matrix[0].len());
        println!("  Objects: {} labels, {} attrs", obj_labels.len(), attr_labels.len());
    }

    #[test]
    fn test_formal_context_antihydra() {
        let tm = antihydra();
        let (matrix, obj_labels, attr_labels) = build_trace_formal_context(&tm, 1000, 3, 10);
        println!("  Objects: {} ({} labels)", matrix.len(), obj_labels.len());
        println!("  Attributes: {} ({} labels)", matrix[0].len(), attr_labels.len());
        assert!(matrix.len() > 0);
        assert!(matrix[0].len() > 0);
    }

    #[test]
    fn test_lattice_antihydra_small() {
        let tm = antihydra();
        let (matrix, _, _) = build_trace_formal_context(&tm, 1000, 2, 10);
        let lattice = build_lattice(&matrix, 500, 30.0);
        println!("  L=2: {} concepts, {} edges", lattice.concepts.len(), lattice.edges.len());
        assert!(lattice.concepts.len() > 0);
    }

    #[test]
    fn test_full_pipeline_antihydra_l1() {
        let tm = antihydra();
        let params = DynamicsParams::uniform();
        let (matrix, _, _) = build_trace_formal_context(&tm, 1000, 1, 10);
        let lattice = build_lattice(&matrix, 500, 30.0);
        let (results, stats) = run_lattice_analysis(&lattice, &params);
        let rho = extract_top_rho(&results, &stats);
        println!("  Antihydra L=1: ρ_top = {:?}", rho);
        assert!(rho.is_some());
        assert!(rho.unwrap() > 0.0 && rho.unwrap() < 1.0);
    }

    #[test]
    fn test_full_pipeline_champion_l1() {
        let tm = current_champion();
        let params = DynamicsParams::uniform();
        let (matrix, _, _) = build_trace_formal_context(&tm, 1000, 1, 10);
        let lattice = build_lattice(&matrix, 500, 30.0);
        let (results, stats) = run_lattice_analysis(&lattice, &params);
        let rho = extract_top_rho(&results, &stats);
        println!("  Champion L=1: ρ_top = {:?}", rho);
        assert!(rho.is_some());
    }

    #[test]
    fn test_simple_halter_vs_nonhalter() {
        let params = DynamicsParams::uniform();

        let tm_halt = simple_halter();
        let (m_h, _, _) = build_trace_formal_context(&tm_halt, 100, 2, 1);
        let l_h = build_lattice(&m_h, 500, 30.0);
        let (r_h, s_h) = run_lattice_analysis(&l_h, &params);
        let rho_h = extract_top_rho(&r_h, &s_h);

        let tm_non = simple_nonhalter();
        let (m_n, _, _) = build_trace_formal_context(&tm_non, 100, 2, 1);
        let l_n = build_lattice(&m_n, 500, 30.0);
        let (r_n, s_n) = run_lattice_analysis(&l_n, &params);
        let rho_n = extract_top_rho(&r_n, &s_n);

        println!("  Simple Halter L=2:    ρ_top = {:?}", rho_h);
        println!("  Simple NonHalter L=2: ρ_top = {:?}", rho_n);
        if let (Some(rh), Some(rn)) = (rho_h, rho_n) {
            assert!(rh > 0.0 && rh < 1.0);
            assert!(rn > 0.0 && rn < 1.0);
        }
    }

    #[test]
    fn test_transition_graph_context() {
        let tm = antihydra();
        let (matrix, obj_labels, attr_labels) = build_transition_graph_context(&tm);
        println!("  Transition graph: {} objects, {} attrs", matrix.len(), matrix[0].len());
        println!("  Objects: {:?}", obj_labels);
        println!("  Attributes: {:?}", attr_labels);
        let lattice = build_lattice(&matrix, 500, 30.0);
        println!("  Concepts: {}, edges: {}", lattice.concepts.len(), lattice.edges.len());
        assert!(lattice.concepts.len() > 2);
    }

    #[test]
    fn test_rho_sensitivity_to_lattice_size() {
        // Test if ρ_top varies with lattice size for chain lattices
        let params = DynamicsParams::uniform();
        println!("\n  ρ_top sensitivity to lattice size (chain lattices):");
        println!("  {:>8} {:>12} {:>12}", "Size", "Concepts", "ρ_top");
        println!("  {:>8} {:>12} {:>12}", "----", "--------", "------");

        for n in [2, 3, 4, 5, 10, 20, 50, 100] {
            let lattice = fca::build_chain_lattice(n);
            let stats = pipeline::compute_lattice_stats(&lattice);
            let results = pipeline::run_topological_iteration(&lattice, &stats, &params);
            let rho = extract_top_rho(&results, &stats);
            println!("  {:>8} {:>12} {:>12.8}", n, lattice.concepts.len(),
                rho.unwrap_or(f64::NAN));
        }
    }

    #[test]
    fn test_rho_sensitivity_to_diamond_lattice() {
        let params = DynamicsParams::uniform();
        let lattice = fca::build_diamond_lattice();
        let stats = pipeline::compute_lattice_stats(&lattice);
        let results = pipeline::run_topological_iteration(&lattice, &stats, &params);
        let rho = extract_top_rho(&results, &stats);
        println!("\n  Diamond lattice: {} concepts, ρ_top = {:?}", lattice.concepts.len(), rho);
    }
}