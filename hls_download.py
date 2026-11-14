import earthaccess

auth = earthaccess.login(strategy="netrc")
if not auth.authenticated:
    raise SystemExit("Вход не выполнен, проверь _netrc")

results = earthaccess.search_data(
    short_name="HLSL30",
    version="2.0",
    bounding_box=(-112.07088, 32.87118, -111.88474, 33.0262),
    temporal=("2021-04-01", "2021-10-31"),
)

results = [r for r in results if "T12SVB" in r["umm"]["GranuleUR"]]
print(len(results), "granules after tile filter")
for r in results:
    print(r["umm"]["GranuleUR"])

if len(results) > 30:
    raise SystemExit("Слишком много гранул, проверь фильтр")

earthaccess.download(results, local_path=r"D:\carver-data\hls")
