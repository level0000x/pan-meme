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