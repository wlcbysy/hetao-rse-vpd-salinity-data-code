# Push Instructions

This folder is already initialized as a local git repository on the `main`
branch. You can verify the current final commit with:

```bash
git log --oneline -1
```

To upload it to GitHub, create an empty public repository on GitHub, then run
the following commands from this folder:

```bash
git remote add origin https://github.com/<your-user-or-org>/hetao-rse-vpd-salinity-data-code.git
git push -u origin main
```

After the push succeeds, replace the manuscript data-availability placeholder:

```text
[GitHub repository URL to be inserted after upload]
```

with the final repository URL.
