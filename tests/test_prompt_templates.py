from meeting_minutes.prompt_templates import (
    format_special_terms,
    load_saved_templates,
    prompt_variables,
    render_prompt,
    save_templates,
)


def test_render_prompt_replaces_known_values_and_keeps_unknown_placeholders():
    rendered = render_prompt(
        "Contexte : {contexte}; transcript : {transcript}; inconnu : {autre}",
        {"contexte": "Projet OMS", "transcript": "Bonjour"},
    )

    assert rendered == "Contexte : Projet OMS; transcript : Bonjour; inconnu : {autre}"


def test_prompt_variables_preserves_first_use_order():
    assert prompt_variables("{transcript} {contexte} {transcript}") == [
        "transcript",
        "contexte",
    ]


def test_format_special_terms_ignores_empty_lines():
    assert format_special_terms("OneStock\n\n Chausséa ") == "- OneStock\n- Chausséa"


def test_saved_templates_round_trip(tmp_path):
    path = tmp_path / "saved_prompts.json"
    save_templates(path, {"Mes notes": "Voici {transcript}"})

    assert load_saved_templates(path) == {"Mes notes": "Voici {transcript}"}
