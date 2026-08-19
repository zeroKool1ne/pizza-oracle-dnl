We need to install the dependencies locally:

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

Then, we need to generate the data:
```
python src/generate_data.py --era normal --rows 1000 --out orders.csv
head -5 orders.csv
```

---

Then, upload the file to our registry:
```
aws s3 cp orders.csv s3://pizza-oracle-<yourname>/data/orders.csv
```
