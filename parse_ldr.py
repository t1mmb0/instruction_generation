def parse_ldr(file_path):
    parts = []
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens and tokens[0] == '1':  # '1' = Teilzeile
                color = int(tokens[1])
                x, y, z = map(float, tokens[2:5])
                part_id = tokens[-1]
                parts.append({
                    "part_id": part_id,
                    "color": color,
                    "x": x,
                    "y": y,
                    "z": z
                })
    return parts