#!/usr/bin/env python3
with open('/Users/didi/Desktop/GBGCL/tools/sweepX.py', 'rb') as f:
    raw = f.read()

idx = raw.find(b'FILTERS = {')
end = raw.find(b'}', idx)
block = raw[idx:end+2]

new_block = (
    b'FILTERS = {\n'
    b'    "Photo": [\n'
    b'        ("detach","cos",0.3),    # P1: cos similarity\n'
    b'        ("edges","cos",0.3),     # P2: edges quality + cos\n'
    b'        ("homo","cos",0.5),     # P3: homo + alpha=0.5\n'
    b'        ("detach","dot",0.5),   # P4: alpha=0.5 intermediate\n'
    b'        ("edges","dot",0.3),     # P5: edges + dot\n'
    b'    ],\n'
    b'    "Computers": [\n'
    b'        ("edges","cos",0.3),   # P1: edges + cos\n'
    b'        ("homo","cos",0.5),    # P2: homo + cos + alpha=0.5\n'
    b'        ("detach","cos",0.5),   # P3: detach + cos + alpha=0.5\n'
    b'        ("edges","dot",0.5),   # P4: edges + dot + alpha=0.5\n'
    b'        ("detach","cos",0.3),   # P5: detach + cos baseline\n'
    b'        ("detach","dot",0.7),    # baseline control: original best config\n'
    b'        ("homo","dot",0.7),     # baseline control\n'
    b'    ],\n'
    b'    "Physics": [\n'
    b'        ("detach","cos",0.3),   # P1: detach + cos\n'
    b'        ("edges","dot",0.3),    # P2: edges + dot\n'
    b'        ("homo","cos",0.3),    # baseline\n'
    b'        ("detach","dot",0.3),   # baseline\n'
    b'    ],\n'
    b'    "CS": [\n'
    b'        # CS already beats baseline, resume from checkpoint\n'
    b'        ("detach","dot",0.7),\n'
    b'        ("detach","dot",0.3),\n'
    b'    ],\n'
    b'}\n'
)

new_raw = raw[:idx] + new_block + raw[end+2:]
with open('/Users/didi/Desktop/GBGCL/tools/sweepX.py', 'wb') as f:
    f.write(new_raw)
print('OK')