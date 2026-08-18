/*
 * ========================================================================
 * NOVA — REAL NEURAL NETWORK (FINAL STABLE VERSION WITH CONFUSION MATRIX)
 * ========================================================================
 * Complete features:
 * - Safe memory allocation with strict dimension validation
 * - Fixed Backpropagation pipeline with ReLU derivative
 * - Mini-Batch Stochastic Gradient Descent (SGD)
 * - Binary weight saving and loading persistence
 * - Native PGM (P2/P5) file reader for real 28x28 human-written digits
 * - Fully automated ASCII Confusion Matrix evaluation
 * ========================================================================
 * Compile: gcc -O3 -o ai code.c -lm
 * Run: ./ai
 * ========================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define INPUT_SIZE 784 // 28x28 pixels
#define H1 256
#define H2 128
#define OUT 10
#define LR 0.02f        // Stabilized learning rate
#define BATCH_SIZE 32
#define EPOCHS 15
#define TRAIN 10000     // Matches up to 10k dataset limit
#define TEST 2000

typedef struct { int rows, cols; float* d; } Mat;

Mat* m_create(int r, int c) {
    Mat* m = calloc(1, sizeof(Mat));
    if (!m) return NULL;
    m->rows = r; m->cols = c;
    m->d = calloc(r * c, sizeof(float));
    if (!m->d) { free(m); return NULL; }
    return m;
}

void m_free(Mat* m) { if (m) { free(m->d); free(m); } }

void m_rand(Mat* m, float s) {
    if (!m || !m->d) return;
    for (int i = 0; i < m->rows * m->cols; i++)
        m->d[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * s;
}

void m_zero(Mat* m) { if (m && m->d) memset(m->d, 0, m->rows * m->cols * sizeof(float)); }

void m_copy(Mat* d, Mat* s) { if (d && s && d->d && s->d) memcpy(d->d, s->d, s->rows * s->cols * sizeof(float)); }

void m_mul(Mat* a, Mat* b, Mat* o) {
    if (!a || !b || !o || !a->d || !b->d || !o->d) return;
    if (a->cols != b->rows || o->rows != a->rows || o->cols != b->cols) return;
    for (int i = 0; i < o->rows; i++) {
        for (int j = 0; j < o->cols; j++) {
            float s = 0;
            for (int k = 0; k < a->cols; k++)
                s += a->d[i * a->cols + k] * b->d[k * b->cols + j];
            o->d[i * o->cols + j] = s;
        }
    }
}

void m_add(Mat* a, Mat* b, Mat* o) {
    if (!a || !b || !o || !a->d || !b->d || !o->d) return;
    for (int i = 0; i < a->rows * a->cols; i++) o->d[i] = a->d[i] + b->d[i];
}

void m_Tmul_A(Mat* a, Mat* b, Mat* o) {
    if (!a || !b || !o || !a->d || !b->d || !o->d) return;
    if (a->rows != b->rows || o->rows != a->cols || o->cols != b->cols) return;
    for (int i = 0; i < o->rows; i++) {
        for (int j = 0; j < o->cols; j++) {
            float s = 0;
            for (int k = 0; k < a->rows; k++)
                s += a->d[k * a->cols + i] * b->d[k * b->cols + j];
            o->d[i * o->cols + j] = s;
        }
    }
}

void m_Tmul_B(Mat* a, Mat* b, Mat* o) {
    if (!a || !b || !o || !a->d || !b->d || !o->d) return;
    if (a->cols != b->cols || o->rows != a->rows || o->cols != b->rows) return;
    for (int i = 0; i < o->rows; i++) {
        for (int j = 0; j < o->cols; j++) {
            float s = 0;
            for (int k = 0; k < a->cols; k++)
                s += a->d[i * a->cols + k] * b->d[j * b->cols + k];
            o->d[i * o->cols + j] = s;
        }
    }
}

void m_relu(Mat* m) { if (m && m->d) for (int i = 0; i < m->rows * m->cols; i++) m->d[i] = m->d[i] > 0 ? m->d[i] : 0; }

void m_softmax(Mat* m) {
    if (!m || !m->d) return;
    float max = m->d[0];
    for (int i = 1; i < m->rows * m->cols; i++) if (m->d[i] > max) max = m->d[i];
    float sum = 0;
    for (int i = 0; i < m->rows * m->cols; i++) { m->d[i] = expf(m->d[i] - max); sum += m->d[i]; }
    if (sum > 0) for (int i = 0; i < m->rows * m->cols; i++) m->d[i] /= sum;
}

typedef struct { 
    Mat *w, *b; 
    Mat *dw, *db; 
    Mat *out, *in, *delta; 
    char act[16]; 
} Layer;

void layer_free(Layer* l) { 
    if (l) { 
        m_free(l->w); m_free(l->b); 
        m_free(l->dw); m_free(l->db);
        m_free(l->out); m_free(l->in); m_free(l->delta); 
        free(l); 
    } 
}

Layer* layer_new(int in, int out, const char* act) {
    Layer* l = calloc(1, sizeof(Layer));
    if (!l) return NULL;
    l->w = m_create(out, in);
    l->b = m_create(out, 1);
    l->dw = m_create(out, in);
    l->db = m_create(out, 1);
    l->out = m_create(out, 1);
    l->in = m_create(in, 1);
    l->delta = m_create(out, 1);
    if (!l->w || !l->b || !l->dw || !l->db || !l->out || !l->in || !l->delta) { layer_free(l); return NULL; }
    m_rand(l->w, sqrtf(2.0f / in));
    m_zero(l->b);
    strncpy(l->act, act, 15);
    return l;
}

void layer_forward(Layer* l, Mat* input) {
    if (!l || !input) return;
    m_copy(l->in, input);
    m_mul(l->w, input, l->out);
    m_add(l->out, l->b, l->out);
    if (strcmp(l->act, "relu") == 0) m_relu(l->out);
}

void layer_backward(Layer* l, Mat* upstream_grad) {
    if (!l || !upstream_grad) return;
    
    m_copy(l->delta, upstream_grad);
    if (strcmp(l->act, "relu") == 0) {
        for (int i = 0; i < l->delta->rows * l->delta->cols; i++) {
            if (l->out->d[i] <= 0.0f) l->delta->d[i] = 0.0f;
        }
    }

    Mat* wg = m_create(l->w->rows, l->w->cols);
    if (!wg) return;
    m_Tmul_B(l->delta, l->in, wg);
    
    for (int i = 0; i < l->w->rows * l->w->cols; i++) l->dw->d[i] += wg->d[i];
    for (int i = 0; i < l->b->rows * l->b->cols; i++) l->db->d[i] += l->delta->d[i];
    
    m_free(wg);
}

void layer_update(Layer* l, int batch_size) {
    if (!l) return;
    float scale = LR / (float)batch_size;
    for (int i = 0; i < l->w->rows * l->w->cols; i++) l->w->d[i] -= l->dw->d[i] * scale;
    for (int i = 0; i < l->b->rows * l->b->cols; i++) l->b->d[i] -= l->db->d[i] * scale;
    m_zero(l->dw);
    m_zero(l->db);
}

typedef struct { Layer* l1, * l2, * l3; float loss, acc; int epoch; } NN;

void nn_free(NN* n) { if (n) { layer_free(n->l1); layer_free(n->l2); layer_free(n->l3); free(n); } }

NN* nn_new() {
    NN* n = calloc(1, sizeof(NN));
    if (!n) return NULL;
    n->l1 = layer_new(INPUT_SIZE, H1, "relu");
    n->l2 = layer_new(H1, H2, "relu");
    n->l3 = layer_new(H2, OUT, "none");
    if (!n->l1 || !n->l2 || !n->l3) { nn_free(n); return NULL; }
    return n;
}

void nn_forward(NN* n, Mat* in) {
    if (!n || !in) return;
    layer_forward(n->l1, in);
    layer_forward(n->l2, n->l1->out);
    layer_forward(n->l3, n->l2->out);
    m_softmax(n->l3->out);
}

void nn_backward(NN* n, int label) {
    if (!n) return;
    Mat* g3 = m_create(OUT, 1);
    if (!g3) return;
    
    for (int i = 0; i < OUT; i++) g3->d[i] = n->l3->out->d[i];
    g3->d[label] -= 1.0f;
    
    layer_backward(n->l3, g3);
    
    Mat* g2 = m_create(H2, 1);
    if (g2) {
        m_Tmul_A(n->l3->w, n->l3->delta, g2);
        layer_backward(n->l2, g2);
        
        Mat* g1 = m_create(H1, 1);
        if (g1) {
            m_Tmul_A(n->l2->w, n->l2->delta, g1);
            layer_backward(n->l1, g1);
            m_free(g1);
        }
        m_free(g2);
    }
    m_free(g3);
}

void nn_update(NN* n, int batch_size) {
    if (!n) return;
    layer_update(n->l1, batch_size);
    layer_update(n->l2, batch_size);
    layer_update(n->l3, batch_size);
}

int nn_predict(NN* n) {
    if (!n || !n->l3 || !n->l3->out) return 0;
    int b = 0;
    for (int i = 1; i < OUT; i++) if (n->l3->out->d[i] > n->l3->out->d[b]) b = i;
    return b;
}

void nn_save(NN* n, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) { printf("Erro ao salvar pesos.\n"); return; }
    fwrite(n->l1->w->d, sizeof(float), n->l1->w->rows * n->l1->w->cols, f);
    fwrite(n->l1->b->d, sizeof(float), n->l1->b->rows * n->l1->b->cols, f);
    fwrite(n->l2->w->d, sizeof(float), n->l2->w->rows * n->l2->w->cols, f);
    fwrite(n->l2->b->d, sizeof(float), n->l2->b->rows * n->l2->b->cols, f);
    fwrite(n->l3->w->d, sizeof(float), n->l3->w->rows * n->l3->w->cols, f);
    fwrite(n->l3->b->d, sizeof(float), n->l3->b->rows * n->l3->b->cols, f);
    fclose(f);
    printf("📊 Pesos salvos com sucesso em '%s'!\n", filename);
}

bool nn_load(NN* n, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return false;
    size_t r = 0;
    r += fread(n->l1->w->d, sizeof(float), n->l1->w->rows * n->l1->w->cols, f);
    r += fread(n->l1->b->d, sizeof(float), n->l1->b->rows * n->l1->b->cols, f);
    r += fread(n->l2->w->d, sizeof(float), n->l2->w->rows * n->l2->w->cols, f);
    r += fread(n->l2->b->d, sizeof(float), n->l2->b->rows * n->l2->b->cols, f);
    r += fread(n->l3->w->d, sizeof(float), n->l3->w->rows * n->l3->w->cols, f);
    r += fread(n->l3->b->d, sizeof(float), n->l3->b->rows * n->l3->b->cols, f);
    fclose(f);
    return r > 0;
}

bool load_pgm(const char* filename, float* buffer, int expected_size) {
    FILE* f = fopen(filename, "rb");
    if (!f) return false;

    char header[16];
    if (!fgets(header, sizeof(header), f)) { fclose(f); return false; }
    if (header[0] != 'P' || (header[1] != '2' && header[1] != '5')) { fclose(f); return false; }

    int ch = fgetc(f);
    while (ch == '#') {
        while ((ch = fgetc(f)) != '\n' && ch != EOF);
        ch = fgetc(f);
    }
    ungetc(ch, f);

    int width, height, max_val;
    if (fscanf(f, "%d %d %d", &width, &height, &max_val) != 3) { fclose(f); return false; }
    fgetc(f); 

    if (width * height != expected_size) {
        printf("Erro: %s tem dimensoes erradas (%dx%d). Esperado 28x28.\n", filename, width, height);
        fclose(f);
        return false;
    }

    if (header[1] == '5') {
        unsigned char* temp = malloc(expected_size);
        if (temp) {
            size_t rb = fread(temp, 1, expected_size, f);
            if ((int)rb == expected_size) {
                for (int i = 0; i < expected_size; i++) buffer[i] = (float)temp[i] / 255.0f;
            }
            free(temp);
        }
    } else {
        int val;
        for (int i = 0; i < expected_size; i++) {
            if (fscanf(f, "%d", &val) != 1) { 
                fclose(f); 
                return false; 
            }
            buffer[i] = (float)val / 255.0f;
        }
    }

    fclose(f);
    return true;
}

typedef struct {
    float** imgs;
    int* labels;
    int count, size;
} Dataset;

Dataset* ds_new(int count, int size) {
    Dataset* d = calloc(1, sizeof(Dataset));
    if (!d) return NULL;
    
    d->imgs = calloc(count, sizeof(float*));
    d->labels = calloc(count, sizeof(int));
    if (!d->imgs || !d->labels) {
        free(d->imgs);
        free(d->labels);
        free(d);
        return NULL;
    }
    
    d->count = count;
    d->size = size;
    for (int i = 0; i < count; i++) {
        d->imgs[i] = calloc(size, sizeof(float));
        if (!d->imgs[i]) {
            for (int j = 0; j < i; j++) free(d->imgs[j]);
            free(d->imgs);
            free(d->labels);
            free(d);
            return NULL;
        }
    }
    return d;
}

void ds_free(Dataset* d) {
    if (d) {
        if (d->imgs) {
            for (int i = 0; i < d->count; i++) free(d->imgs[i]);
            free(d->imgs);
        }
        free(d->labels);
        free(d);
    }
}

bool ds_load_real(Dataset* d, const char* base_path) {
    if (!d) return false;
    char path[256];
    int loaded = 0;
    
    for (int label = 0; label < OUT; label++) {
        int items_per_class = d->count / OUT;
        for (int i = 0; i < items_per_class; i++) {
            sprintf(path, "%s/%d/%d.pgm", base_path, label, i);
            if (load_pgm(path, d->imgs[loaded], d->size)) {
                d->labels[loaded] = label;
                loaded++;
            }
        }
    }
    
    if (loaded == 0) {
        printf("⚠️ Nenhuma imagem PGM encontrada em '%s'. Usando fallback artificial para nao quebrar.\n", base_path);
        for (int i = 0; i < d->count; i++) {
            int lbl = rand() % OUT;
            d->labels[i] = lbl;
            for (int j = 0; j < d->size; j++) {
                d->imgs[i][j] = ((float)rand() / RAND_MAX) * 0.1f;
            }
            for (int j = 0; j < 20; j++) {
                d->imgs[i][(lbl * 20 + j) % d->size] = 1.0f;
            }
        }
        return true;
    }
    
    printf("✅ Carregadas %d imagens reais de '%s'\n", loaded, base_path);
    d->count = loaded;
    return true;
}

float nn_accuracy(NN* n, Dataset* d) {
    if (!n || !d || d->count == 0) return 0;
    Mat* in = m_create(INPUT_SIZE, 1);
    if (!in) return 0;
    
    int correct = 0;
    for (int i = 0; i < d->count; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            in->d[j] = d->imgs[i][j];
        }
        nn_forward(n, in);
        if (nn_predict(n) == d->labels[i]) correct++;
    }
    m_free(in);
    return (float)correct / d->count;
}

void nn_confusion_matrix(NN* n, Dataset* d) {
    if (!n || !d || d->count == 0) return;
    int matrix[OUT][OUT] = {{0}};
    Mat* in = m_create(INPUT_SIZE, 1);
    if (!in) return;
    
    for (int i = 0; i < d->count; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            in->d[j] = d->imgs[i][j];
        }
        nn_forward(n, in);
        int pred = nn_predict(n);
        int target = d->labels[i];
        matrix[target][pred]++;
    }
    m_free(in);
    
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                 📊 MATRIZ DE CONFUSÃO (TESTE)                  ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf(" Real \\ Prev ->\n");
    printf("       ");
    for (int j = 0; j < OUT; j++) printf("[%d]  ", j);
    printf("\n ─────────────────────────────────────────────────────────────\n");
    
    for (int i = 0; i < OUT; i++) {
        printf("  [%d] │ ", i);
        for (int j = 0; j < OUT; j++) {
            if (i == j) {
                if (matrix[i][j] > 0) printf("\033[1;32m%3d\033[0m  ", matrix[i][j]); // Green
                else printf("%3d  ", matrix[i][j]);
            } else {
                if (matrix[i][j] > 0) printf("\033[1;31m%3d\033[0m  ", matrix[i][j]); // Red
                else printf("%3d  ", matrix[i][j]);
            }
        }
        printf("\n");
    }
    printf(" ─────────────────────────────────────────────────────────────\n\n");
}

void nn_train(NN* n, Dataset* train, Dataset* test) {
    if (!n || !train || !test || train->count == 0) return;
    
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  NOVA — REAL NEURAL NETWORK TRAINING                          ║\n");
    printf("║  ════════════════════════════════════════════════════════════  ║\n");
    printf("║  Training: %d samples  |  Testing: %d samples                ║\n", train->count, test->count);
    printf("║  Learning Rate: %.3f  |  Batch Size: %d                    ║\n", LR, BATCH_SIZE);
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    
    Mat* in = m_create(INPUT_SIZE, 1);
    if (!in) return;
    
    for (int e = 0; e < EPOCHS; e++) {
        n->epoch = e + 1;
        float loss = 0;
        
        for (int i = train->count - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            float* tmp = train->imgs[i];
            train->imgs[i] = train->imgs[j];
            train->imgs[j] = tmp;
            
            int tl = train->labels[i];
            train->labels[i] = train->labels[j];
            train->labels[j] = tl;
        }
        
        int accumulated = 0;
        for (int i = 0; i < train->count; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                in->d[j] = train->imgs[i][j];
            }
            nn_forward(n, in);
            loss += -logf(n->l3->out->d[train->labels[i]] + 1e-8f);
            nn_backward(n, train->labels[i]);
            accumulated++;
            
            if (accumulated == BATCH_SIZE || i == train->count - 1) {
                nn_update(n, accumulated);
                accumulated = 0;
            }
        }
        
        loss /= train->count;
        n->loss = loss;
        
        float ta = nn_accuracy(n, train);
        float tea = nn_accuracy(n, test);
        n->acc = tea;
        
        printf("Epoch %2d/%d | Loss: %.6f | Train Acc: %.2f%% | Test Acc: %.2f%%\n",
               e + 1, EPOCHS, loss, ta * 100, tea * 100);
    }
    m_free(in);
}

int main() {
    srand((unsigned)time(NULL));
    NN* nn = nn_new();
    if (!nn) return 1;
    
    printf("\n🔮 Verificando se existe modelo previamente treinado...\n");
    if (nn_load(nn, "nova_pesos.bin")) {
        printf("🎉 Pesos carregados do arquivo 'nova_pesos.bin'. Pulando treino!\n");
    } else {
        printf("📂 Sem pesos salvos. Inicializando carregamento e treino...\n");
        Dataset* train = ds_new(TRAIN, INPUT_SIZE);
        Dataset* test = ds_new(TEST, INPUT_SIZE);
        
        ds_load_real(train, "data/train");
        ds_load_real(test, "data/test");
        
        nn_train(nn, train, test);
        nn_save(nn, "nova_pesos.bin");
        nn_confusion_matrix(nn, test);
        
        ds_free(train);
        ds_free(test);
    }
    
    printf("\n🔮 Interactive Mode (Real Image Loader)\n");
    printf("  Digite o caminho de uma imagem PGM de 28x28 pixels para testar\n");
    printf("  Exemplo: meu_numero.pgm\n");
    printf("  Digite 'quit' para sair\n\n");
    
    Mat* in = m_create(INPUT_SIZE, 1);
    char cmd[256];
    
    while (1) {
        printf("Caminho da Imagem PGM: ");
        fflush(stdout);
        
        if (!fgets(cmd, sizeof(cmd), stdin)) break;
        cmd[strcspn(cmd, "\n")] = '\0';
        
        if (strcmp(cmd, "quit") == 0 || strcmp(cmd, "exit") == 0) break;
        
        if (!load_pgm(cmd, in->d, INPUT_SIZE)) {
            printf("❌ Erro ao abrir ou processar o arquivo '%s'. Verifique se eh um PGM valido de 28x28.\n", cmd);
            continue;
        }
        
        nn_forward(nn, in);
        int pred = nn_predict(nn);
        
        printf("\n✅ RESULTADO DO SEU DESENHO:\n");
        printf("  Previsao da Rede: %d (Confianca: %.2f%%)\n", pred, nn->l3->out->d[pred] * 100);
        printf("  Probabilities:\n");
        for (int i = 0; i < OUT; i++) {
            printf("    [%d]: %.2f%%\n", i, nn->l3->out->d[i] * 100);
        }
        printf("\n");
    }
    
    m_free(in);
    nn_free(nn);
    return 0;
}
