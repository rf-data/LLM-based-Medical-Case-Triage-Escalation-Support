        def cached_batch(texts: list[str]) -> list[dict]:
            results = [None] * len(texts)
            missing_texts = []
            missing_idx = []

            for i, text in enumerate(texts):
                key = make_cache_key(text, 
                                 prompt, 
                                 namespace)
                cached = load_from_cache(key)
                if cached is not None:
                    results[i] = cached
                else:
                    missing_texts.append(text)
                    missing_idx.append(i)

            if missing_texts:
                fresh = batch_escalation_by_llm(
                    texts=missing_texts,
                    prompt=prompt,
                    scheme=scheme,
                    allowed_values=allowed_values,
                )

                for idx, res in zip(missing_idx, fresh):
                    key = make_cache_key(texts[idx], 
                                        prompt, 
                                        namespace)
                    save_to_cache(key, res)
                    results[idx] = res

            return results
