prefix = /usr/local
exec_prefix = ${prefix}
libdir = ${exec_prefix}/lib
includedir = ${prefix}/include

CXXFLAGS = -g
LIBS = -lmesquite
LDFLAGS = -L$(libdir)
CPPFLAGS = -I$(includedir)

run: test_wrapper.cpp
run: test_lowlevel.cpp
#run: test_laplacian_smoothing.cpp
	$(CXX) -o $@ $< $(CPPFLAGS) $(LDFLAGS) $(LIBS)

clean:
	rm -f run 
