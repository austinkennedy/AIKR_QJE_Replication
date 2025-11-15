rm(list=ls())
options(scipen=999)

options(repos = c(CRAN = "https://cloud.r-project.org"))
if (!require("pak", quietly = TRUE)) install.packages("pak")

pkgs <- c(
    "car@3.1.2",
    "data.table@1.15.0",
    "fixest@0.11.2",
    "ggplot2@3.5.0",
    "ggpubr@0.6.0",
    "margins@0.3.26",
    "sandwich@3.1.0",
    "scales@1.3.0",
    "yaml@2.3.8",
    "ggtern@3.5.0",
    "tidyverse@2.0.0",
    "biscale@1.1.0",
    "cowplot@1.1.3",
    "modelsummary@1.4.5"
)

for (spec in pkgs) {
  pkg <- sub("@.*$", "", spec)      # extract package name
  ver <- sub(".*@", "", spec)       # extract version
  if (!requireNamespace(pkg, quietly = TRUE) ||
      packageVersion(pkg) != ver) {
    pak::pkg_install(spec, ask = FALSE)
  }
}