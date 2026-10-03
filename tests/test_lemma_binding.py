from __future__ import annotations

import unittest

from metrics_rebuild.share.lemma_binding import (
    bind_function_contexts,
    build_clause_wrapper,
    render_clause_expression,
    render_clause_wrapper,
)


def _context(
    parameters: list[dict[str, str]],
    returns: list[dict[str, str]] | None = None,
    *,
    mode: str = "exec",
    generic_parameters: str = "",
    where_clause: str = "",
) -> dict:
    return {
        "parameters": parameters,
        "returns": returns or [],
        "mode": mode,
        "generic_parameters": generic_parameters,
        "where_clause": where_clause,
    }


class TestLemmaBinding(unittest.TestCase):
    def test_names_are_bound_by_position_to_canonical_variables(self) -> None:
        reference = _context(
            [{"name": "x", "type": "int"}, {"name": "flag", "type": "bool"}],
            [{"name": "result", "type": "int"}],
        )
        generated = _context(
            [{"name": "n", "type": "int"}, {"name": "enabled", "type": "bool"}],
            [{"name": "res", "type": "int"}],
        )

        result = bind_function_contexts(reference, generated)

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.plan)
        plan = result.plan
        assert plan is not None
        self.assertEqual(
            plan.parameter_dicts(),
            [
                {"name": "__arg0", "type": "int"},
                {"name": "__arg1", "type": "bool"},
                {"name": "__ret0", "type": "int"},
            ],
        )
        self.assertEqual(plan.inputs[0].reference_name, "x")
        self.assertEqual(plan.inputs[0].generated_name, "n")
        self.assertEqual(plan.returns[0].canonical_name, "__ret0")
        self.assertTrue(plan.has_source_renames)

    def test_parameter_count_mismatch_is_structured_unknown(self) -> None:
        result = bind_function_contexts(
            _context([{"name": "x", "type": "int"}]),
            _context(
                [
                    {"name": "x", "type": "int"},
                    {"name": "y", "type": "int"},
                ]
            ),
        )

        self.assertFalse(result.ok)
        assert result.issue is not None
        self.assertEqual(
            result.issue.to_dict(),
            {
                "holds": None,
                "status": "unknown",
                "reason": "target_signature_mismatch",
                "category": "parameter_count",
                "reference": 1,
                "generated": 2,
            },
        )

        unnamed = bind_function_contexts(
            _context([{"name": None, "type": "int"}]),
            _context([{"name": "x", "type": "int"}]),
        )
        assert unnamed.issue is not None
        self.assertEqual(unnamed.issue.category, "parameter_name")

    def test_return_count_and_position_type_mismatches_are_rejected(self) -> None:
        return_count = bind_function_contexts(
            _context([], [{"name": "out", "type": "int"}]),
            _context([]),
        )
        assert return_count.issue is not None
        self.assertEqual(return_count.issue.category, "return_count")

        input_type = bind_function_contexts(
            _context([{"name": "x", "type": "Seq<int>"}]),
            _context([{"name": "n", "type": "Set<int>"}]),
        )
        assert input_type.issue is not None
        self.assertEqual(input_type.issue.category, "parameter_type")
        self.assertEqual(input_type.issue.position, 0)

    def test_whitespace_only_type_differences_are_compatible(self) -> None:
        result = bind_function_contexts(
            _context([{"name": "x", "type": "Map<int, Seq<bool>>"}]),
            _context([{"name": "m", "type": "Map < int, Seq < bool > >"}]),
        )
        self.assertTrue(result.ok)

    def test_lifetime_only_type_differences_bind_with_reference_type(self) -> None:
        result = bind_function_contexts(
            _context([{"name": "type_spec", "type": "&str"}], [{"name": "result", "type": "DType"}]),
            _context([{"name": "spec", "type": "&'static str"}], [{"name": "result", "type": "DType"}]),
        )
        self.assertTrue(result.ok)
        assert result.plan is not None
        self.assertEqual(result.plan.parameter_dicts()[0], {"name": "__arg0", "type": "&str"})

        wrapper = build_clause_wrapper(
            result.plan,
            {"source": "generated", "kind": "ensures", "text": "spec == \"int8\" ==> result.itemsize == 1"},
            wrapper_name="__sqm_lifetime",
        )
        assert wrapper.wrapper is not None
        self.assertIn("spec: &str", render_clause_wrapper(wrapper.wrapper))

        mutable = bind_function_contexts(
            _context([{"name": "v", "type": "&mut Vec<u8>"}]),
            _context([{"name": "v", "type": "&'a mut Vec<u8>"}]),
        )
        self.assertTrue(mutable.ok)

        different = bind_function_contexts(
            _context([{"name": "s", "type": "&str"}]),
            _context([{"name": "s", "type": "&'static String"}]),
        )
        assert different.issue is not None
        self.assertEqual(different.issue.category, "parameter_type")

    def test_unnamed_returns_bind_by_position_and_get_collision_free_names(self) -> None:
        result = bind_function_contexts(
            _context(
                [{"name": "__lemma_ret_0", "type": "int"}],
                [{"name": None, "type": "Vec<int>"}],
            ),
            _context(
                [{"name": "x", "type": "int"}],
                [{"name": "result", "type": "Vec<int>"}],
            ),
        )
        self.assertTrue(result.ok)
        assert result.plan is not None
        self.assertEqual(result.plan.returns[0].reference_name, "__lemma_ret_0_2")
        self.assertEqual(result.plan.returns[0].generated_name, "result")
        self.assertEqual(result.plan.returns[0].canonical_name, "__ret0")

        wrapper = build_clause_wrapper(
            result.plan,
            {
                "source": "reference",
                "kind": "ensures",
                "text": "__lemma_ret_0_2.len() >= 0",
            },
            wrapper_name="__sqm_unnamed_return",
        )
        assert wrapper.wrapper is not None
        self.assertIn("__lemma_ret_0_2: Vec<int>", render_clause_wrapper(wrapper.wrapper))

    def test_tuple_tail_comma_normalization_is_conservative(self) -> None:
        compatible = bind_function_contexts(
            _context([], [{"name": None, "type": "(usize,\n usize,)"}]),
            _context([], [{"name": "out", "type": "(usize, usize)"}]),
        )
        self.assertTrue(compatible.ok)

        singleton = bind_function_contexts(
            _context([], [{"name": None, "type": "(T,)"}]),
            _context([], [{"name": None, "type": "T"}]),
        )
        parenthesized = bind_function_contexts(
            _context([], [{"name": None, "type": "(T)"}]),
            _context([], [{"name": None, "type": "T"}]),
        )
        assert singleton.issue is not None
        assert parenthesized.issue is not None
        self.assertEqual(singleton.issue.category, "return_type")
        self.assertEqual(parenthesized.issue.category, "return_type")

        malformed_tuple = bind_function_contexts(
            _context([], [{"name": None, "type": "(T,,)"}]),
            _context([], [{"name": None, "type": "(T,)"}]),
        )
        assert malformed_tuple.issue is not None
        self.assertEqual(malformed_tuple.issue.category, "return_type")

    def test_mode_and_generic_context_are_conservative(self) -> None:
        mode = bind_function_contexts(_context([], mode="exec"), _context([], mode="spec"))
        assert mode.issue is not None
        self.assertEqual(mode.issue.category, "mode")

        generic = bind_function_contexts(
            _context(
                [{"name": "x", "type": "T"}],
                generic_parameters="<T: Copy>",
                where_clause="where T: Eq,",
            ),
            _context(
                [{"name": "x", "type": "T"}],
                generic_parameters="<T>",
                where_clause="where T: Eq + Copy,",
            ),
        )
        assert generic.issue is not None
        self.assertEqual(generic.issue.reason, "generic_context_mismatch")

        formatting_only = bind_function_contexts(
            _context(
                [{"name": "x", "type": "T"}],
                generic_parameters="<T: Copy>",
                where_clause="where T: Eq,",
            ),
            _context(
                [{"name": "n", "type": "T"}],
                generic_parameters="< T : Copy >",
                where_clause="where   T : Eq ,",
            ),
        )
        self.assertTrue(formatting_only.ok)

    def test_mutable_reference_has_current_and_old_snapshots(self) -> None:
        result = bind_function_contexts(
            _context(
                [
                    {"name": "values", "type": "&mut Vec<int>"},
                    {"name": "nested", "type": "&'a mut Vec<Vec<T>>"},
                ],
                generic_parameters="<'a, T>",
            ),
            _context(
                [
                    {"name": "xs", "type": "&mut Vec<int>"},
                    {"name": "ys", "type": "&'a mut Vec<Vec<T>>"},
                ],
                generic_parameters="<'a, T>",
            ),
        )
        self.assertTrue(result.ok)
        assert result.plan is not None
        self.assertEqual(
            result.plan.parameter_dicts(),
            [
                {"name": "__arg0", "type": "&Vec<int>"},
                {"name": "__old_arg0", "type": "&Vec<int>"},
                {"name": "__arg1", "type": "&'a Vec<Vec<T>>"},
                {"name": "__old_arg1", "type": "&'a Vec<Vec<T>>"},
            ],
        )

    def test_wrappers_preserve_source_names_and_bind_at_call_site(self) -> None:
        binding = bind_function_contexts(
            _context(
                [{"name": "x", "type": "int"}],
                [{"name": "result", "type": "int"}],
            ),
            _context(
                [{"name": "n", "type": "int"}],
                [{"name": "res", "type": "int"}],
            ),
        )
        assert binding.plan is not None

        generated = build_clause_wrapper(
            binding.plan,
            {
                "source": "generated",
                "kind": "ensures",
                "text": "forall|x: int| x >= n ==> res >= n",
            },
            wrapper_name="__sqm_gen_clause",
        )
        reference = build_clause_wrapper(
            binding.plan,
            {
                "source": "reference",
                "kind": "ensures",
                "text": "result >= x",
            },
            wrapper_name="__sqm_ref_clause",
        )

        assert generated.wrapper is not None
        assert reference.wrapper is not None
        self.assertEqual(
            generated.wrapper.body_text,
            "forall|x: int| x >= n ==> res >= n",
        )
        self.assertEqual(generated.wrapper.call, "__sqm_gen_clause(__arg0, __ret0)")
        self.assertEqual(reference.wrapper.call, "__sqm_ref_clause(__arg0, __ret0)")
        self.assertIn("n: int", render_clause_wrapper(generated.wrapper))
        self.assertIn("res: int", render_clause_wrapper(generated.wrapper))
        self.assertNotIn("__arg0 >=", render_clause_wrapper(generated.wrapper))

    def test_mutable_wrapper_uses_old_for_precondition_and_current_for_postcondition(self) -> None:
        binding = bind_function_contexts(
            _context([{"name": "values", "type": "&mut Vec<int>"}]),
            _context([{"name": "xs", "type": "&mut Vec<int>"}]),
        )
        assert binding.plan is not None

        precondition = build_clause_wrapper(
            binding.plan,
            {
                "_lemma_source": "generated",
                "kind": "requires",
                "text": "xs@.len() > 0",
            },
            wrapper_name="__sqm_pre",
        )
        postcondition = build_clause_wrapper(
            binding.plan,
            {
                "source": "generated",
                "kind": "ensures",
                "text": "xs@.len() == old(xs)@.len() + 1 && old(*xs)@.len() >= 0",
            },
            wrapper_name="__sqm_post",
        )

        assert precondition.wrapper is not None
        assert postcondition.wrapper is not None
        self.assertEqual(precondition.wrapper.call, "__sqm_pre(__old_arg0)")
        self.assertEqual(
            postcondition.wrapper.call,
            "__sqm_post(__arg0, __old_arg0)",
        )
        self.assertEqual(
            postcondition.wrapper.body_text,
            "xs@.len() == __old_xs@.len() + 1 && (*__old_xs)@.len() >= 0",
        )
        self.assertIn("xs: &Vec<int>", render_clause_wrapper(postcondition.wrapper))
        self.assertIn("__old_xs: &Vec<int>", render_clause_wrapper(postcondition.wrapper))

    def test_old_rewrite_does_not_touch_methods_or_ordinary_identifiers(self) -> None:
        binding = bind_function_contexts(
            _context([{"name": "x", "type": "int"}]),
            _context([{"name": "x", "type": "int"}]),
        )
        assert binding.plan is not None
        wrapper = build_clause_wrapper(
            binding.plan,
            {
                "source": "reference",
                "kind": "ensures",
                "text": "old(x) == x && helper::old(x) && obj.old(x)",
            },
            wrapper_name="__sqm_clause",
        )
        assert wrapper.wrapper is not None
        self.assertEqual(
            wrapper.wrapper.body_text,
            "x == x && helper::old(x) && obj.old(x)",
        )

    def test_unknown_origin_is_rejected_only_when_names_differ(self) -> None:
        renamed = bind_function_contexts(
            _context([{"name": "x", "type": "int"}]),
            _context([{"name": "n", "type": "int"}]),
        )
        assert renamed.plan is not None
        ambiguous = build_clause_wrapper(
            renamed.plan,
            {"kind": "requires", "text": "x > 0"},
            wrapper_name="__sqm_clause",
        )
        assert ambiguous.issue is not None
        self.assertEqual(ambiguous.issue.reason, "ambiguous_clause_origin")

        same_names = bind_function_contexts(
            _context([{"name": "x", "type": "int"}]),
            _context([{"name": "x", "type": "int"}]),
        )
        assert same_names.plan is not None
        inferred = build_clause_wrapper(
            same_names.plan,
            {"kind": "requires", "text": "x > 0"},
            wrapper_name="__sqm_clause",
        )
        self.assertTrue(inferred.ok)
        assert inferred.wrapper is not None
        self.assertEqual(inferred.wrapper.source, "reference")

    def test_control_flow_expression_renderer_only_wraps_top_level_logical_operands(self) -> None:
        leading_if = "if flag { left } else { right } && ready"
        rhs_match = "ready ==> match value { Some(x) => x > 0, None => false }"
        self.assertEqual(
            render_clause_expression(leading_if),
            "(if flag { left } else { right }) && ready",
        )
        self.assertEqual(
            render_clause_expression(rhs_match),
            "ready ==> (match value { Some(x) => x > 0, None => false })",
        )
        self.assertEqual(
            render_clause_expression("(if flag { left } else { right }) && ready"),
            "(if flag { left } else { right }) && ready",
        )
        self.assertEqual(
            render_clause_expression("helper(if flag { left } else { right })"),
            "helper(if flag { left } else { right })",
        )


if __name__ == "__main__":
    unittest.main()
