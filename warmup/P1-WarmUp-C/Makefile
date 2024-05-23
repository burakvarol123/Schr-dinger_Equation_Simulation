all: warmup

MOD = modules
INC = -I modules -I /opt/homebrew/opt/gsl/include

MACROS=-O3 #-DCG_DBG

$(MOD)/%.o: $(MOD)/%.c $(MOD)/%.h
	gcc -c -o $@ $< $(INC) -Wall $(MACROS)

$(MOD)/%.o: $(MOD)/%.c
	gcc -c -o $@ $< $(INC) -Wall $(MACROS)

%.o: %.c
	gcc -c -o $@ $< $(INC) -Wall $(MACROS)

warmup: warmup.o $(MOD)/utils.o $(MOD)/linalg.o $(MOD)/conjgrad.o $(MOD)/conjgrad_test.o $(MOD)/powermethod.o $(MOD)/powermethod_test.o
	gcc -o $@ $^ $(INC) -Wall -lm -lgsl -lgslcblas -L /opt/homebrew/opt/gsl/lib $(MACROS)

clean:
	rm -f $(MOD)/*.o *.o warmup
