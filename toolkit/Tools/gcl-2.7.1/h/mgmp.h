#ifdef __SHORT_LIMB
typedef unsigned int		mp_limb_t;
#else
#ifdef __LONG_LONG_LIMB
typedef unsigned long long int	mp_limb_t;
#else
typedef unsigned long int	mp_limb_t;
#endif
#endif

typedef mp_limb_t *		mp_ptr;

typedef struct
{
  int _mp_alloc;		/* Number of *limbs* allocated and pointed
				   to by the _mp_d field.  */
  int _mp_size;			/* abs(_mp_size) is the number of limbs the
				   last field points to.  If _mp_size is
				   negative this is a negative number.  */
  mp_limb_t *_mp_d;		/* Pointer to the limbs.  */
} __mpz_struct;

typedef __mpz_struct MP_INT;
typedef __mpz_struct * mpz_t;

/* Available random number generation algorithms.  */
typedef enum
{
  GMP_RAND_ALG_DEFAULT = 0,
  GMP_RAND_ALG_LC = GMP_RAND_ALG_DEFAULT /* Linear congruential.  */
} gmp_randalg_t;

/* Linear congruential data struct.  */
typedef struct {
  mpz_t _mp_a;			/* Multiplier. */
  unsigned long int _mp_c;	/* Adder. */
  mpz_t _mp_m;			/* Modulus (valid only if m2exp == 0).  */
  unsigned long int _mp_m2exp;	/* If != 0, modulus is 2 ^ m2exp.  */
} __gmp_randata_lc;

/* Random state struct.  */
typedef struct
{
  mpz_t _mp_seed;		/* Current seed.  */
  gmp_randalg_t _mp_alg;	/* Algorithm used.  */
  union {			/* Algorithm specific data.  */
    __gmp_randata_lc *_mp_lc;	/* Linear congruential.  */
  } _mp_algdata;
} __gmp_randstate_struct;
typedef __gmp_randstate_struct gmp_randstate_t[1];

#define mpz_sgn(x_)  ((x_)->_mp_size < 0 ? -1 : (x_)->_mp_size > 0)
#define mpz_odd_p(x_)  (((x_)->_mp_size != 0) & ((int) ((x_)->_mp_d[0])))
#define mpz_even_p(x_)  (! (((x_)->_mp_size != 0) & ((int) ((x_)->_mp_d[0]))))
