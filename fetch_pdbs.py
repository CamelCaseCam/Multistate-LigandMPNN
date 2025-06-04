from rcsbapi.search import AttributeQuery, TextQuery
from rcsbapi.search import search_attributes as attrs

# Construct a "full-text" sub-query for structures associated with the term "Hemoglobin"
q_nmr = AttributeQuery(
    attribute="rcsb_entry_info.experimental_method",
    operator="exact_match",
    value="NMR",
)

q_models = AttributeQuery(
    attribute="rcsb_entry_info.deposited_model_count",
    operator="greater",
    value=2,
)

q_protein = AttributeQuery(
    attribute="entity_poly.rcsb_entity_polymer_type",
    operator="exact_match",
    value="Protein",
)

# Combine the sub-queries (can sub-group using parentheses and standard operators, "&", "|", etc.)
query = q_nmr & q_models & q_protein

# Fetch the results by iterating over the query execution
out = ""
for rId in query():
    out += f"{rId}\n"
out = out.splitlines()

# Take the final 100 as test data
out_train = out[:-100]
out_test = out[-100:]

# Write the results to a file
with open("training/nmr_train.txt", "w") as f:
    for line in out_train:
        f.write(f"{line}\n")

with open("training/nmr_test.txt", "w") as f:
    for line in out_test:
        f.write(f"{line}\n")