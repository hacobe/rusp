import itertools
import string
import sys


if __name__ == "__main__":
	var_to_values = {}
	templates = []
	for i in range(1, len(sys.argv)):
		if sys.argv[i].startswith("--"):
			var, values_str = sys.argv[i].replace("--", "").split("=")
			values = values_str.split(",")
			var_to_values[var] = values
		else:
			templates.append(sys.argv[i])

	outputs = []
	for template in templates:
		var_list = []
		for _, v, _, _ in string.Formatter().parse(template):
			if v is None:
				continue
			var_list.append(v)

		if not var_list:
			outputs.append(template)
			continue

		values_list = []
		for var in var_list:
			values_list.append(var_to_values[var])
		for prod in itertools.product(*values_list):
			prod_dict = {}
			for i, value in enumerate(prod):
				prod_dict[var_list[i]] = value
			output = template.format(**prod_dict)
			outputs.append(output)

	print(" ".join(outputs))

