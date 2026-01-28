def batch_escalation_by_llm(
    texts: list[str],
    prompt: str,
    scheme: dict,
    allowed_values: str,
) -> list[dict]:
    llm_model = session.llm_model
    if llm_model == "openai":
        client = get_openai_client()

    content = content_creator_batch(texts, scheme, allowed_values)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
    )

    data = json.loads(response.choices[0].message.content)

    if not isinstance(data, list):
        raise TypeError(f"Batch LLM must return list[dict], got {type(data)}")

    return [normalize_llm_response(d) for d in data]
