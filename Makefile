# @ means do not echo to screen the command
# -C change to directory
all: 
	@$(MAKE) -C KMeans --no-print-directory

clean:
	@cd KMeans && $(MAKE) clean --no-print-directory && cd .. 