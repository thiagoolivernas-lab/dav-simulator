import wfdb

record = wfdb.rdrecord(
    "3000003",
    pn_dir="mimic3wdb/30/3000003"
)

print(record.sig_name)
print(record.fs)