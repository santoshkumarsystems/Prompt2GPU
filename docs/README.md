# Prompt2GPU interactive dashboard

**Author:** Santosh Kumar
**Version:** 1.0

`index.html` is a **committed, already-generated** dashboard containing the real Prompt2GPU V1.0 measurements collected on **Google Colab / NVIDIA Tesla T4**. A viewer does not need Python, CUDA, vLLM, or an NVIDIA GPU to open it.

The page is generated from the committed CSV/JSON evidence by `scripts/build_dashboard.py`. Rebuilding is optional and is only needed after intentionally changing benchmark result files.

## Local viewing — zero install

Open `docs/index.html` directly, or from the repository root run:

```bash
make serve
```

Then open `http://localhost:8000`.

## GitHub Pages

After pushing the repository:

1. Open **Settings → Pages**.
2. Choose **Deploy from a branch**.
3. Select branch **main** and folder **/docs**.
4. Save.

GitHub will provide the Pages URL. Static charts in the root README work even if Pages is not enabled.
