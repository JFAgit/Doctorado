p <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/GSE30595_M1_2026/06_datasets_externos/GMCSF_basales_busqueda/metadata/GSE269009_series_matrix.txt.gz"
lines <- readLines(gzfile(p))

parse_row <- function(row) {
  vals <- strsplit(sub("^[^\t]+\t", "", row), "\t")[[1]]
  gsub("^\"|\"$", "", vals)
}

title <- parse_row(lines[grep("^!Sample_title", lines)[1]])
chars <- lines[grep("^!Sample_characteristics_ch1", lines)]
cell <- sub("cell type: ", "", parse_row(chars[1]))
tb <- sub("tb status: ", "", parse_row(chars[2]))
sex <- sub("Sex: ", "", parse_row(chars[3]))
d <- data.frame(title = title, cell = cell, tb = tb, sex = sex)
print(table(d$cell, d$tb, d$sex))
print(table(d$tb[d$cell == "MP"], d$sex[d$cell == "MP"]))
