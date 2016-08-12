TARGET = opencv_test
CC = g++ 
LIBDIR =  "C:\Users\lab309\Downloads\opencv\build\install\x86\mingw\bin"
INCLUDES = "C:\Users\lab309\Downloads\opencv\build\install\include" 
LIBS = opencv_core310 opencv_ffmpeg310 opencv_world310

 .PHONY: all

 all: $(TARGET)

 $(TARGET):
	$(CC) opencv_test.cpp -o $(TARGET) $(addprefix -L, $(LIBDIR)) $(addprefix -I, $(INCLUDES)) $(addprefix -l, $(LIBS))