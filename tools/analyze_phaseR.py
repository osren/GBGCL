import csv

def stats(path, dataset, num_epochs, use_gb, quity, sim, alpha):
    with open(path) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    h = {name: i for i, name in enumerate(header)}
    matching = []
    for r in rows[1:]:
        if len(r) < len(header):
            continue
        try:
            if (r[h['dataset']] == dataset
                and int(r[h['num_epochs']]) == num_epochs
                and int(float(r[h['use_gb']])) == use_gb
                and (quity is None or r[h['gb_quity']] == quity)
                and (sim is None or r[h['gb_sim']] == sim)
                and (alpha is None or float(r[h['gb_alpha']]) == alpha)):
                matching.append(r)
        except (ValueError, IndexError):
            pass
    accs = [float(r[h['clf_mean']]) for r in matching]
    return accs, matching

print(f"{'Group':<40} {'n':<3} {'mean':<8} {'min':<8} {'max':<8}")
print('-' * 70)

groups = [
    ('R-A Photo baseline (use_gb=0)',      'Photo', 700, 0, None, None, None),
    ('R-B Photo Top-1 (detach,dot,0.3)',   'Photo', 700, 1, 'detach', 'dot', 0.3),
    ('R-C Computers baseline (use_gb=0)',  'Computers', 700, 0, None, None, None),
    ('R-D Computers Top-1 (homo,dot,0.7)', 'Computers', 700, 1, 'homo', 'dot', 0.7),
]
results = {}
for label, *args in groups:
    path = 'results/phaseR/' + args[0] + '_summary.csv'
    accs, _ = stats(path, *args)
    results[label] = accs
    if accs:
        mean = sum(accs) / len(accs)
        print(f"{label:<40} {len(accs):<3} {mean:<8.4f} {min(accs):<8.4f} {max(accs):<8.4f}")
    else:
        print(f"{label:<40} -- no rows")

print()
print("=== Delta vs SGRL baselines ===")
print(f"{'Group':<40} {'R mean':<8} {'SGRL':<8} {'Δ':<8}")
print('-' * 70)
sgrl = {'Photo': 0.9395, 'Computers': 0.9023}
for label, accs in results.items():
    if accs:
        ds = 'Photo' if 'Photo' in label else 'Computers'
        mean = sum(accs) / len(accs)
        d = mean - sgrl[ds]
        print(f"{label:<40} {mean:<8.4f} {sgrl[ds]:<8.4f} {d:+.4f}")

print()
print("=== Top-1 vs Baseline delta (within Phase R) ===")
photo_top1 = sum(results['R-B Photo Top-1 (detach,dot,0.3)']) / len(results['R-B Photo Top-1 (detach,dot,0.3)'])
photo_base = sum(results['R-A Photo baseline (use_gb=0)']) / len(results['R-A Photo baseline (use_gb=0)'])
comp_top1 = sum(results['R-D Computers Top-1 (homo,dot,0.7)']) / len(results['R-D Computers Top-1 (homo,dot,0.7)'])
comp_base = sum(results['R-C Computers baseline (use_gb=0)']) / len(results['R-C Computers baseline (use_gb=0)'])
print(f"Photo:    Top-1 {photo_top1:.4f} vs Baseline {photo_base:.4f}, Δ = {photo_top1-photo_base:+.4f}")
print(f"Computers: Top-1 {comp_top1:.4f} vs Baseline {comp_base:.4f}, Δ = {comp_top1-comp_base:+.4f}")