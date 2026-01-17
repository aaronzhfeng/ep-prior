# Makefile for EP-Prior paper

PAPER = ep_prior
LATEX = pdflatex
BIBTEX = bibtex

# Default target
all: $(PAPER).pdf

# Build PDF
$(PAPER).pdf: $(PAPER).tex $(PAPER).bib
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

# Quick build (no bibliography update)
quick:
	$(LATEX) $(PAPER)

# Clean auxiliary files
clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).toc $(PAPER).lof $(PAPER).lot

# Clean everything including PDF
distclean: clean
	rm -f $(PAPER).pdf

# Word count (approximate)
wordcount:
	@detex $(PAPER).tex | wc -w

# Check for common issues
check:
	@echo "Checking for common issues..."
	@grep -n "TODO" $(PAPER).tex || echo "No TODOs found"
	@grep -n "PLACEHOLDER" $(PAPER).tex || echo "No PLACEHOLDERs found"
	@grep -n "XXX" $(PAPER).tex || echo "No XXXs found"

.PHONY: all quick clean distclean wordcount check

