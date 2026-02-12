def postprocess(texts):
  results = []

  for t in texts:
    t = t.strip()
    t.replace(" ", "")
    t.replace("_", "")

    results.append(t)

  return results