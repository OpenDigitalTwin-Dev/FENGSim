/* Copyright (C) 2024 Camm Maguire */
/*
	x : fixnum or bignum (may be not normalized)
	y : integer
   returns
	fixnum or bignum ( not normalized )
*/

object big_log_op();
object normalize_big(object);

static fixnum
fixnum_log_op2(fixnum op,fixnum x,fixnum y) {
  
  return fixnum_boole(op,x,y);

}

static object
integer_log_op2(fixnum op,object x,enum type tx,object y,enum type ty) {

  object u=big_fixnum1;
  object ux=tx==t_bignum ? x : (mpz_set_si(MP(big_fixnum2),fix(x)), big_fixnum2);
  object uy=ty==t_bignum ? y : (mpz_set_si(MP(big_fixnum3),fix(y)), big_fixnum3);
  
  switch(op) {
  case BOOLCLR:	 mpz_set_si(MP(u),0);break;
  case BOOLSET:	 mpz_set_si(MP(u),-1);break;
  case BOOL1:	 mpz_set(MP(u),MP(ux));break;
  case BOOL2:	 mpz_set(MP(u),MP(uy));break;
  case BOOLC1:	 mpz_com(MP(u),MP(ux));break;
  case BOOLC2:	 mpz_com(MP(u),MP(uy));break;
  case BOOLAND:	 mpz_and(MP(u),MP(ux),MP(uy));break;
  case BOOLIOR:	 mpz_ior(MP(u),MP(ux),MP(uy));break;
  case BOOLXOR:	 mpz_xor(MP(u),MP(ux),MP(uy));break;
  case BOOLEQV:	 mpz_xor(MP(u),MP(ux),MP(uy));mpz_com(MP(u),MP(u));break;
  case BOOLNAND: mpz_and(MP(u),MP(ux),MP(uy));mpz_com(MP(u),MP(u));break;
  case BOOLNOR:	 mpz_ior(MP(u),MP(ux),MP(uy));mpz_com(MP(u),MP(u));break;
  case BOOLANDC1:mpz_com(MP(u),MP(ux));mpz_and(MP(u),MP(u),MP(uy));break;
  case BOOLANDC2:mpz_com(MP(u),MP(uy));mpz_and(MP(u),MP(ux),MP(u));break;
  case BOOLORC1: mpz_com(MP(u),MP(ux));mpz_ior(MP(u),MP(u),MP(uy));break;
  case BOOLORC2: mpz_com(MP(u),MP(uy));mpz_ior(MP(u),MP(ux),MP(u));break;
  default:break;/*FIXME error*/
  }
    
  return u;

}

object
log_op2(fixnum op,object x,object y) {

  enum type tx=type_of(x),ty=type_of(y);

  if (tx==t_fixnum && ty==t_fixnum)
    return make_fixnum(fixnum_log_op2(op,fix(x),fix(y)));
  else
    return maybe_replace_big(integer_log_op2(op,x,tx,y,ty));
}


static int
big_bitp(object x, ufixnum p)
{
  return mpz_tstbit(MP(x),p);
}

static int
mpz_bitcount(__mpz_struct *x)
{
  if (mpz_sgn(x) >= 0) {
    return mpz_popcount(x);
  } else {
    object u = new_bignum();
    mpz_com(MP(u),x);
    return mpz_popcount(MP(u));
  }
}


static int
mpz_bitlength(__mpz_struct *x)
{
  if (mpz_sgn(x) >= 0) {
    return mpz_sizeinbase(x,2);
  } else {
    object u = new_bignum();
    mpz_com(MP(u),x);
    return mpz_sizeinbase(MP(u),2);
  }
}



