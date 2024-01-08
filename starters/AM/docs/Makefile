BOOK=measure_report

all:
	xelatex am
	bibtex am
	xelatex am
	xelatex am

#final:
#	xelatex $(BOOK)
#	biber $(BOOK)
#	makeindex $(BOOK)
#	xelatex $(BOOK)
#	xelatex $(BOOK)

clean:
	rm -f *.aux *.mtc* *.out *.idx *.bcf *.log *~ *.ptc *.toc *.run.xml *.maf
