# ========================================================================
# NOVA ENGINE AUTOMATED BUILD SYSTEM
# ========================================================================

CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c99
LIBS = -lm
TARGET = ai
SOURCE = code.c
PESOS = nova_pesos.bin

.PHONY: all run clean extract test-reset

# Default rule: Compiles the engine
all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "🚀 Compiling NOVA Engine under aggressive optimization (-O3)..."
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET) $(LIBS)
	@echo "✅ Compilation successful. Executable generated: ./$(TARGET)"

# Setup workspace and run the application
run: all
	@if [ ! -d "data" ]; then \
		echo "📂 Data directory missing. Extracting dataset first..."; \
		make extract; \
	fi
	./$(TARGET)

# Extract dataset using the local Python utility
extract:
	@echo "📦 Unpacking local MNIST raw compressed streams..."
	python3 extract_local_mnist.py

# Force reset memory weights and drop persistence cache before training
test-reset:
	@echo "🧹 Dropping weight binary persistence cache..."
	rm -f $(PESOS)
	@make run

# Cleanup compiled files and artifacts
clean:
	@echo "🧹 Cleaning workspace artifacts..."
	rm -f $(TARGET) $(PESOS)
	@echo "✨ Workspace cleaned."
