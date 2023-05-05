

paper.pdf: paper.tex $(wildcard *.tex) bibliography.bib
	pdflatex $<
	bibtex paper
	pdflatex $<
	pdflatex $<

