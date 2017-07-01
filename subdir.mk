################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../CSVM.cpp \
../cnpy.cpp \
../error.cpp \
../libsvm.cpp \
../main.cpp \
../path.cpp 

OBJS += \
./CSVM.o \
./cnpy.o \
./error.o \
./libsvm.o \
./main.o \
./path.o 

CPP_DEPS += \
./CSVM.d \
./cnpy.d \
./error.d \
./libsvm.d \
./main.d \
./path.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O3 -Wall -c -fmessage-length=0 -std=c++11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


