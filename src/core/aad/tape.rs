//! The AAD tape (Wengert list) and the backward sweep.
//!
//! Every arithmetic operation on [`Var`](super::var::Var) records one
//! node holding the indices of its (up to two) parents and the local
//! partial derivatives with respect to them. The backward sweep walks
//! the tape once in reverse, accumulating adjoints — so the gradient of
//! one scalar output with respect to **every** input costs one forward
//! evaluation plus one reverse pass, independent of the number of
//! inputs. That O(1) property is the whole point for Greeks.

use std::cell::RefCell;

/// Sentinel parent index for leaf slots.
const NONE: usize = usize::MAX;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Node {
    pub(crate) parents: [usize; 2],
    pub(crate) partials: [f64; 2],
}

/// The recording tape. Create one per differentiated computation (or
/// [`clear`](Tape::clear) between computations, e.g. per Monte Carlo
/// path).
#[derive(Debug, Default)]
pub struct Tape {
    pub(crate) nodes: RefCell<Vec<Node>>,
}

impl Tape {
    pub fn new() -> Tape {
        Tape::default()
    }

    /// A new independent input variable.
    pub fn var(&self, value: f64) -> super::var::Var<'_> {
        let idx = self.push0();
        super::var::Var { tape: self, idx, val: value }
    }

    /// Number of recorded nodes.
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.borrow().is_empty()
    }

    /// Drop all recorded nodes (existing `Var`s become invalid).
    pub fn clear(&self) {
        self.nodes.borrow_mut().clear();
    }

    pub(crate) fn push0(&self) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        nodes.push(Node { parents: [NONE, NONE], partials: [0.0, 0.0] });
        nodes.len() - 1
    }

    pub(crate) fn push1(&self, parent: usize, partial: f64) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        nodes.push(Node { parents: [parent, NONE], partials: [partial, 0.0] });
        nodes.len() - 1
    }

    pub(crate) fn push2(&self, p0: usize, w0: f64, p1: usize, w1: f64) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        nodes.push(Node { parents: [p0, p1], partials: [w0, w1] });
        nodes.len() - 1
    }

    /// Backward sweep from the node `output`: returns the adjoint of
    /// every node, i.e. `d output / d node`.
    pub(crate) fn backward(&self, output: usize) -> Vec<f64> {
        let nodes = self.nodes.borrow();
        let mut adjoint = vec![0.0; nodes.len()];
        adjoint[output] = 1.0;
        for i in (0..=output).rev() {
            let a = adjoint[i];
            if a == 0.0 {
                continue;
            }
            let node = nodes[i];
            for slot in 0..2 {
                let p = node.parents[slot];
                if p != NONE {
                    adjoint[p] += node.partials[slot] * a;
                }
            }
        }
        adjoint
    }
}

/// The result of a backward sweep: query with the input `Var`s.
#[derive(Debug, Clone)]
pub struct Gradients {
    pub(crate) adjoints: Vec<f64>,
}

impl Gradients {
    /// `d output / d v`.
    pub fn wrt(&self, v: super::var::Var<'_>) -> f64 {
        self.adjoints[v.idx]
    }
}
