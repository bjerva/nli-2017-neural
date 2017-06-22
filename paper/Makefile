FILE=bea12nli
LATEX=pdflatex
all:
	${LATEX} ${FILE}
	bibtex ${FILE}
	${LATEX} ${FILE}
	${LATEX} ${FILE}
	${LATEX} ${FILE}
	rm -f *.bbl *.log *.blg *.aux
clean:
	rm -f *.bbl *.log *.blg ${FILE}.pdf *.aux
