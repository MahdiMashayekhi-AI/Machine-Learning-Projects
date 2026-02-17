def postprocess(texts):
  results = []

  for t in texts:
    t = t.strip()
    t = t.replace(" ", "")
    t = t.replace("_", "")

    results.append(t)

  return results