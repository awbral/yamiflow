#include "udf.h"
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

/* dynamic memory allocation for 1D and 2D arrays */
#define DECLARE_MEMORY(name, type) type *name = NULL

#define DECLARE_MEMORY_N(name, type, dim) type *name[dim] = {NULL}

#define RELEASE_MEMORY(name)                                        \
if (NNULLP(name)) {                                                 \
    free(name);                                                     \
    name = NULL;                                                    \
}

#define RELEASE_MEMORY_N(name, dim)                                 \
for (_d = 0; _d < dim; _d++) {                                      \
    RELEASE_MEMORY(name[_d]);                                       \
}

#define ASSIGN_MEMORY(name, size, type)                             \
if (size) {                                                         \
    if (NNULLP(name)) {                                             \
        name = (type *)realloc(name, size * sizeof(type));          \
    } else {                                                        \
        name = (type *)malloc(size * sizeof(type));                 \
    }                                                               \
    if (NULLP(name)) {                                              \
        Error("\nUDF-error: Memory assignment failed for name.");   \
        exit(1);                                                    \
    }                                                               \
}

#define ASSIGN_MEMORY_N(name, size, type, dim)                      \
for (_d = 0; _d < dim; _d++) {                                      \
    ASSIGN_MEMORY(name[_d], size, type);                            \
}

/* sending and receiving arrays in parallel */
#define PRF_CSEND_INT_N(to, name, n, tag, dim)                      \
for (_d = 0; _d < dim; _d++) {                                      \
    PRF_CSEND_INT(to, name[_d], n, tag);                            \
}

#define PRF_CSEND_REAL_N(to, name, n, tag, dim)                     \
for (_d = 0; _d < dim; _d++) {                                      \
    PRF_CSEND_REAL(to, name[_d], n, tag);                           \
}

#define PRF_CRECV_INT_N(from, name, n, tag, dim)                    \
for (_d = 0; _d < dim; _d++) {                                      \
    PRF_CRECV_INT(from, name[_d], n, tag);                          \
}

#define PRF_CRECV_REAL_N(from, name, n, tag, dim)                   \
for (_d = 0; _d < dim; _d++) {                                      \
    PRF_CRECV_REAL(from, name[_d], n, tag);                         \
}

/* global variables */
#define mnpf 4
int _d; /* don't use in UDFs! */
int n_threads;
DECLARE_MEMORY(thread_ids, int);
int velocity = 0;


  /*----------------*/
 /* get_thread_ids */
/*----------------*/

DEFINE_ON_DEMAND(get_thread_ids) {
    /* read in thread thread ids, should be called early on */

#if !RP_NODE
    int k;
    FILE *file;
    file = fopen("bcs.txt", "r");
    fscanf(file, "%i", &n_threads);
#endif /* !RP_NODE */

    host_to_node_int_1(n_threads);
    ASSIGN_MEMORY(thread_ids, n_threads, int);

#if !RP_NODE
    for (k = 0; k < n_threads; k++) {
        fscanf(file, "%*s %i", &thread_ids[k]);
    }
    fclose(file);
#endif /* !RP_NODE */

    host_to_node_int(thread_ids, n_threads);
}



  /*----------------------*/
 /* store_coordinates_id */
/*----------------------*/

DEFINE_ON_DEMAND(store_coordinates_id) {
    if (myid == 0) {printf("\n\nStarted UDF store_coordinates_id.\n"); fflush(stdout);}

    int thread, n_nodes, n_faces, i_n, i_f, d;
    DECLARE_MEMORY_N(node_coords, real, ND_ND);
    DECLARE_MEMORY(node_ids, int);

#if !RP_HOST
    Domain *domain;
    Thread *face_thread;
    face_t face;
    Node *node;
    int node_number;
#endif /* !RP_HOST */

#if !RP_NODE
    char file_nodes_name[256];
    FILE *file_nodes = NULL;
#endif /* !RP_NODE */

#if PARALLEL
    int compute_node;
#endif /* PARALLEL */

    for (thread=0; thread<n_threads; thread++) {

#if !RP_NODE
        sprintf(file_nodes_name, "post_processing/nodes_thread%i.dat", thread_ids[thread]);

        if (NULLP(file_nodes = fopen(file_nodes_name, "w"))) {
            Error("\nUDF-error: Unable to open %s for writing\n", file_nodes_name);
            exit(1);
        }

#if RP_2D
        fprintf(file_nodes, "%27s %27s %10s\n", "x-coordinate", "y-coordinate", "unique-id");
#else /* RP_2D */
        fprintf(file_nodes, "%27s %27s %27s %10s\n", "x-coordinate", "y-coordinate", "z-coordinate", "unique-id");
#endif /* RP_2D */
#endif /* !RP_NODE */

#if !RP_HOST
        domain = Get_Domain(1);
        face_thread = Lookup_Thread(domain, thread_ids[thread]);

        n_nodes = 0;
        begin_f_loop(face, face_thread) {
            n_nodes += F_NNODES(face, face_thread);
        } end_f_loop(face, face_thread)
        n_faces = THREAD_N_ELEMENTS_INT(face_thread);

        ASSIGN_MEMORY_N(node_coords, n_nodes, real, ND_ND);
        ASSIGN_MEMORY(node_ids, n_nodes, int);

        i_n = 0;
        i_f = 0;
        begin_f_loop(face, face_thread) {
            if (i_f >= n_faces) {Error("\nIndex %i >= array size %i.", i_f, n_faces);}

            f_node_loop(face, face_thread, node_number) {
                node = F_NODE(face, face_thread, node_number);

                if (i_n >= n_nodes) {Error("\nIndex %i >= array size %i.", i_n, n_nodes);}
                node_ids[i_n] = NODE_DM_ID(node);
                for (d = 0; d < ND_ND; d++) {
                    node_coords[d][i_n] = NODE_COORD(node)[d];
                }
                i_n++;
            }
            i_f++;
        } end_f_loop(face, face_thread);
#endif /* !RP_HOST */

#if RP_NODE
        compute_node = (I_AM_NODE_ZERO_P) ? node_host : node_zero;

        PRF_CSEND_INT(compute_node, &n_nodes, 1, myid);

        PRF_CSEND_REAL_N(compute_node, node_coords, n_nodes, myid, ND_ND);
        PRF_CSEND_INT(compute_node, node_ids, n_nodes, myid);

        RELEASE_MEMORY_N(node_coords, ND_ND);
        RELEASE_MEMORY(node_ids);

        if(I_AM_NODE_ZERO_P){
            compute_node_loop_not_zero(compute_node) {
                PRF_CRECV_INT(compute_node, &n_nodes, 1, compute_node);

                ASSIGN_MEMORY_N(node_coords, n_nodes, real, ND_ND);
                ASSIGN_MEMORY(node_ids, n_nodes, int);

                PRF_CRECV_REAL_N(compute_node, node_coords, n_nodes, compute_node, ND_ND);
                PRF_CRECV_INT(compute_node, node_ids, n_nodes, compute_node);

                PRF_CSEND_INT(node_host, &n_nodes, 1, compute_node);

                PRF_CSEND_REAL_N(node_host, node_coords, n_nodes, compute_node, ND_ND);
                PRF_CSEND_INT(node_host, node_ids, n_nodes, compute_node);

                RELEASE_MEMORY_N(node_coords, ND_ND);
                RELEASE_MEMORY(node_ids);
            }
        }
#endif /* RP_NODE */

#if RP_HOST
        compute_node_loop(compute_node) {
            PRF_CRECV_INT(node_zero, &n_nodes, 1, compute_node);

            ASSIGN_MEMORY_N(node_coords, n_nodes, real, ND_ND);
            ASSIGN_MEMORY(node_ids, n_nodes, int);

            PRF_CRECV_REAL_N(node_zero, node_coords, n_nodes, compute_node, ND_ND);
            PRF_CRECV_INT(node_zero, node_ids, n_nodes, compute_node);
#endif /* RP_HOST */

#if !RP_NODE
            for (i_n = 0; i_n < n_nodes; i_n++) {
                for (d = 0; d < ND_ND; d++) {
                    fprintf(file_nodes, "%27.17e ", node_coords[d][i_n]);
                }
                fprintf(file_nodes, "%10d\n", node_ids[i_n]);
            }

            RELEASE_MEMORY_N(node_coords, ND_ND);
            RELEASE_MEMORY(node_ids);
#endif /* !RP_NODE */

#if RP_HOST
        } /* close compute_node_loop */
#endif /* RP_HOST */

#if !RP_NODE
        fclose(file_nodes);
#endif /* !RP_NODE */

    } /* close loop over threads */

    if (myid == 0) {printf("\nFinished UDF store_coordinates_id.\n"); fflush(stdout);}
}


  /*-------------------------*/
 /* store_pressure_traction */
/*-------------------------*/


DEFINE_ON_DEMAND(store_pressure_traction) {
    if (myid == 0) {printf("\nStarted UDF store_pressure_traction.\n"); fflush(stdout);}

    int thread, n, i, d;
    DECLARE_MEMORY_N(array, real, 3*ND_ND + 2);
    DECLARE_MEMORY_N(ids, int, mnpf);
#if !RP_HOST
    Domain *domain;
    Thread *face_thread;
    face_t face;
    Node *node;
    int node_number, j;
    real centroid[ND_ND], traction[ND_ND], area[ND_ND];
#endif /* !RP_HOST */

#if !RP_NODE
    char file_name[256];
    FILE *file = NULL;
    velocity = RP_Get_Integer("udf/v");
#endif /* !RP_NODE */

    host_to_node_int_1(velocity);

#if PARALLEL
    int compute_node;
#endif /* PARALLEL */

    for (thread=0; thread<n_threads; thread++) {

#if !RP_NODE
        sprintf(file_name, "post_processing/pressure_traction_v%i_thread%i.dat",
                velocity, thread_ids[thread]);

        if (NULLP(file = fopen(file_name, "w"))) {
            Error("\nUDF-error: Unable to open %s for writing\n", file_name);
            exit(1);
        }

#if RP_2D
        fprintf(file, "%27s %27s %27s %27s %27s %27s %27s %27s %10s\n",
            "x", "y", "area-x", "area-y", "x-shear", "y-shear", "pressure", "gap-face", "unique-ids");
#else /* RP_2D */
        fprintf(file, "%27s %27s %27s %27s %27s %27s %27s %27s %27s %27s %27s %10s\n",
            "x", "y", "z", "area-x", "area-y", "area-z", "x-shear", "y-shear", "z-shear", "pressure", "gap-face", "unique-ids");

#endif /* RP_2D */
#endif /* !RP_NODE */

#if !RP_HOST
        domain = Get_Domain(1);
        face_thread = Lookup_Thread(domain, thread_ids[thread]);

        n = THREAD_N_ELEMENTS_INT(face_thread);

        ASSIGN_MEMORY_N(array, n, real, 3*ND_ND + 2);
        ASSIGN_MEMORY_N(ids, n, int, mnpf);

        i = 0;
        begin_f_loop(face, face_thread) {
            if (i >= n) {Error("\nIndex %i >= array size %i.", i, n);}

            F_CENTROID(centroid, face, face_thread);
            F_AREA(area, face, face_thread);
            NV_VS(traction, =, F_STORAGE_R_N3V(face, face_thread, SV_WALL_SHEAR), *, -1.0);
            for (d = 0; d < ND_ND; d++) {
                array[d][i] = centroid[d];
                array[d+ND_ND][i] = area[d];
                array[d+2*ND_ND][i] = traction[d];
            }
            array[3*ND_ND][i] = F_P(face, face_thread);
            array[3*ND_ND+1][i] = BLOCKED_NARROW_GAP_ALL_FACE_P(face, face_thread) ? 1. : 0.;
            for (j = 0; j < mnpf; j++) {
                ids[j][i] = -1;
            }

            j = 0;
            f_node_loop(face, face_thread, node_number) {
                if (j >= mnpf) {Error("\nIndex %i >= array size %i.", j, mnpf);}
                node = F_NODE(face, face_thread, node_number);
                ids[j][i] = NODE_DM_ID(node);
                j++;
            }
            i++;
        } end_f_loop(face, face_thread);
#endif /* !RP_HOST */

#if RP_NODE
        compute_node = (I_AM_NODE_ZERO_P) ? node_host : node_zero;

        PRF_CSEND_INT(compute_node, &n, 1, myid);

        PRF_CSEND_REAL_N(compute_node, array, n, myid, 3*ND_ND + 2);
        PRF_CSEND_INT_N(compute_node, ids, n, myid, mnpf);

        RELEASE_MEMORY_N(array, 3*ND_ND + 2);
        RELEASE_MEMORY_N(ids, mnpf);

        if(I_AM_NODE_ZERO_P){
            compute_node_loop_not_zero(compute_node) {
                PRF_CRECV_INT(compute_node, &n, 1, compute_node);

                ASSIGN_MEMORY_N(array, n, real, 3*ND_ND + 2);
                ASSIGN_MEMORY_N(ids, n, int, mnpf);

                PRF_CRECV_REAL_N(compute_node, array, n, compute_node, 3*ND_ND + 2);
                PRF_CRECV_INT_N(compute_node, ids, n, compute_node, mnpf);

                PRF_CSEND_INT(node_host, &n, 1, compute_node);

                PRF_CSEND_REAL_N(node_host, array, n, compute_node, 3*ND_ND + 2);
                PRF_CSEND_INT_N(node_host, ids, n, compute_node, mnpf);

                RELEASE_MEMORY_N(array, 3*ND_ND + 2);
                RELEASE_MEMORY_N(ids, mnpf);
            }
        }
#endif /* RP_NODE */

#if RP_HOST
        compute_node_loop(compute_node) {
            PRF_CRECV_INT(node_zero, &n, 1, compute_node);

            ASSIGN_MEMORY_N(array, n, real, 3*ND_ND + 2);
            ASSIGN_MEMORY_N(ids, n, int, mnpf);

            PRF_CRECV_REAL_N(node_zero, array, n, compute_node, 3*ND_ND + 2);
            PRF_CRECV_INT_N(node_zero, ids, n, compute_node, mnpf);
#endif /* RP_HOST */

#if !RP_NODE
            for (i = 0; i < n; i++) {
                for (d = 0; d < 3*ND_ND + 1; d++) {
                    fprintf(file, "%27.17e ", array[d][i]);
                };
                fprintf(file, "%27d ", (int)(array[3*ND_ND + 1][i]));
                for (d = 0; d < mnpf; d++) {
                    fprintf(file, "%10d", ids[d][i]);
                }
                fprintf(file, "\n");
            }
            RELEASE_MEMORY_N(array, 3*ND_ND + 2);
            RELEASE_MEMORY_N(ids, mnpf);
#endif /* !RP_NODE */

#if RP_HOST
        } /* close compute_node_loop */
#endif /* RP_HOST */

#if !RP_NODE
        fclose(file);
#endif /* !RP_NODE */

    } /* close loop over threads */

    if (myid == 0) {printf("\nFinished UDF store_pressure_traction.\n"); fflush(stdout);}
}
