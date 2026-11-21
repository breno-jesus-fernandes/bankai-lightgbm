//! Upstream-compatible C++ code generation for `task=convert_model`.
//!
//! upstream: src/boosting/gbdt_model_text.cpp (v4.7.0), ModelToIfElse / SaveModelToIfElse;
//! src/io/tree.cpp, ToIfElse / NodeToIfElse / NumericalDecisionIfElse / CategoricalDecisionIfElse.
//! The template contains the upstream generator's static C++ scaffolding (MIT).

use std::path::Path;

use lgbm_core::boosting::Gbdt;
use lgbm_core::config::Config;
use lgbm_core::tree::{K_CATEGORICAL_MASK, K_DEFAULT_LEFT_MASK, Tree};
use lgbm_core::utils::format::format_g;

/// Load a model and write the upstream C++ prediction source to the configured output.
pub(super) fn run(cfg: &Config) -> Result<(), String> {
    if cfg.input_model.is_empty() {
        return Err("no model to convert: set input_model=<file>".into());
    }
    let model = Gbdt::from_model_file(Path::new(&cfg.input_model)).map_err(|e| e.to_string())?;
    let generated = render(&model);
    let path = Path::new(&cfg.convert_model);
    // Upstream wraps existing source rather than simply overwriting it. Preserve its exact
    // preprocessor spelling, including #ifndef following a #define of USE_HARD_CODE.
    let text = if path.exists() {
        let original = std::fs::read(path).map_err(|e| e.to_string())?;
        let mut wrapped = b"#define USE_HARD_CODE 0\n#ifndef USE_HARD_CODE\n".to_vec();
        wrapped.extend_from_slice(&original);
        wrapped.extend_from_slice(b"\n#else\n");
        wrapped.extend_from_slice(generated.as_bytes());
        wrapped.extend_from_slice(b"#endif\n");
        wrapped
    } else {
        generated.into_bytes()
    };
    std::fs::write(path, text).map_err(|e| format!("writing converted model: {e}"))
}

fn render(model: &Gbdt) -> String {
    let mut text = include_str!("convert_model_template.cpp").replace("\r\n", "\n");
    for (leaf_index, key) in [(false, "{{RAW_TREES}}"), (true, "{{LEAF_TREES}}")] {
        let mut trees = String::new();
        for (i, tree) in model.trees().iter().enumerate() {
            trees.push_str(&render_tree(tree, i, leaf_index, false));
            trees.push_str(&render_tree(tree, i, leaf_index, true));
            trees.push('\n');
        }
        text = text.replace(key, &trees);
    }
    for (key, suffix) in [
        ("{{RAW_POINTERS}}", ""),
        ("{{MAP_POINTERS}}", "ByMap"),
        ("{{LEAF_POINTERS}}", "Leaf"),
        ("{{LEAF_MAP_POINTERS}}", "LeafByMap"),
    ] {
        let pointers = (0..model.num_trees())
            .map(|i| format!("PredictTree{i}{suffix}"))
            .collect::<Vec<_>>()
            .join(" , ");
        text = text.replace(key, &pointers);
    }
    text
}

fn render_tree(tree: &Tree, index: usize, leaf_index: bool, by_map: bool) -> String {
    let suffix = if leaf_index { "Leaf" } else { "" };
    let map_suffix = if by_map { "ByMap" } else { "" };
    let argument = if by_map {
        "const std::unordered_map<int, double>& arr"
    } else {
        "const double* arr"
    };
    let mut text = format!("double PredictTree{index}{suffix}{map_suffix}({argument}) {{ ");
    if tree.num_leaves <= 1 {
        // Upstream's stump branch uses the stream's default precision (six significant digits)
        // and returns the leaf value even for the leaf-index variant.
        text.push_str(&format!("return {};", format_g(tree.leaf_value[0], 6)));
    } else {
        let cats = tree
            .cat_threshold
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        text.push_str(&format!(
            "const std::vector<uint32_t> cat_threshold = {{{cats}}};double fval = 0.0f; "
        ));
        if tree.num_cat > 0 {
            text.push_str("int int_fval = 0; ");
        }
        render_node(tree, 0, leaf_index, by_map, &mut text);
    }
    text.push_str(" }\n");
    text
}

fn render_node(tree: &Tree, node: i32, leaf_index: bool, by_map: bool, text: &mut String) {
    if node < 0 {
        let leaf = (!node) as usize;
        let value = if leaf_index {
            leaf.to_string()
        } else {
            format_g(tree.leaf_value[leaf], 17)
        };
        text.push_str(&format!("return {value};"));
        return;
    }
    let node = node as usize;
    let feature = tree.split_feature[node];
    if by_map {
        text.push_str(&format!(
            "fval = arr.count({feature}) > 0 ? arr.at({feature}) : 0.0f;"
        ));
    } else {
        text.push_str(&format!("fval = arr[{feature}];"));
    }
    if tree.decision_type[node] & K_CATEGORICAL_MASK != 0 {
        let cat = tree.threshold[node] as usize;
        let start = tree.cat_boundaries[cat];
        let count = tree.cat_boundaries[cat + 1] - start;
        text.push_str(
            "if (std::isnan(fval)) { int_fval = -1; } else { int_fval = static_cast<int>(fval); }",
        );
        text.push_str(&format!("if (int_fval >= 0 && int_fval < 32 * ({count}) && (((cat_threshold[{start} + int_fval / 32] >> (int_fval & 31)) & 1))) {{"));
    } else {
        let missing = (tree.decision_type[node] >> 2) & 3;
        let default_left = tree.decision_type[node] & K_DEFAULT_LEFT_MASK != 0;
        if missing != 2 {
            text.push_str("if (std::isnan(fval)) fval = 0.0;");
        }
        match missing {
            1 => text.push_str(if default_left {
                "if (Tree::IsZero(fval)) {"
            } else {
                "if (!Tree::IsZero(fval)) {"
            }),
            2 => text.push_str(if default_left {
                "if (std::isnan(fval)) {"
            } else {
                "if (!std::isnan(fval)) {"
            }),
            _ => text.push_str(&format!(
                "if (fval <= {}) {{",
                format_g(tree.threshold[node], 17)
            )),
        }
    }
    render_node(tree, tree.left_child[node], leaf_index, by_map, text);
    text.push_str(" } else { ");
    render_node(tree, tree.right_child[node], leaf_index, by_map, text);
    text.push_str(" }");
}
