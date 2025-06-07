static ufixnum
data_pages(void) {

  return page(2*(rb_end-rb_start)+((void *)heap_end-data_start));

}
  
#ifndef NO_FILE_LOCKING

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>

static int pool=-1;
static struct pool {
  ufixnum pid;
  ufixnum n;
  ufixnum s;
} *Pool;
static ufixnum pool_pid,pool_n,pool_s;

static struct flock f,pl,*plp=&pl;
static char gcl_pool[PATH_MAX];

static int
set_lock(void) {
  
  errno=0;
  if (fcntl(pool,F_SETLKW,plp))
    return errno==EINTR ? set_lock() : -1;
  return 0;

}
  
static void
lock_pool(void) {

  pl.l_type=F_WRLCK;
  massert(!set_lock());

}

static void
unlock_pool(void) {

  pl.l_type=F_UNLCK;
  massert(!set_lock());

}

static void
register_pool(int s) {
  lock_pool();
  Pool->n+=s;
  Pool->s+=s*data_pages();
  unlock_pool();
}
  
static void
open_pool(void) {

  if (pool==-1) {

    struct stat ss;
    massert(!lstat(multiprocess_memory_pool,&ss));
    massert(S_ISDIR(ss.st_mode));

    massert(snprintf(gcl_pool,sizeof(gcl_pool),"%s%sgcl_pool",
		     multiprocess_memory_pool,
		     multiprocess_memory_pool[strlen(multiprocess_memory_pool)-1]=='/' ? "" : "/")>=0);
    massert((pool=open(gcl_pool,O_CREAT|O_RDWR,0644))!=-1);
    massert(!ftruncate(pool,sizeof(struct pool)));
    massert((Pool=mmap(NULL,sizeof(struct pool),PROT_READ|PROT_WRITE,MAP_SHARED,pool,0))!=(void *)-1);

    pl.l_type=F_WRLCK;
    pl.l_whence=SEEK_SET;
    pl.l_start=sizeof(Pool->pid);;
    pl.l_len=0;

    f=pl;
    f.l_start=0;
    f.l_len=sizeof(Pool->pid);
    
    if (!fcntl(pool,F_SETLK,&f)) {

      Pool->pid=getpid();

      lock_pool();
      Pool->n=0;
      Pool->s=0;
      unlock_pool();

    }

    f.l_type=F_RDLCK;
    plp=&f;
    massert(!set_lock());

    plp=&pl;

    register_pool(1);
    massert(!atexit(close_pool));

  }

}
#endif

void
close_pool(void) {

#ifndef NO_FILE_LOCKING
  if (pool!=-1) {
    f.l_type=F_WRLCK;
    if (!fcntl(pool,F_SETLK,&f))
      massert(!unlink(gcl_pool) || errno==ENOENT);
    register_pool(-1);
    massert(!close(pool));
    massert(!munmap(Pool,sizeof(struct pool)));
    pool=-1;
  }
#endif
  
}

static void
update_pool(fixnum val) {

#ifndef NO_FILE_LOCKING
  if (multiprocess_memory_pool) {
    open_pool();
    lock_pool();
    Pool->s+=val;
    unlock_pool();
  }
#endif
  
}

static ufixnum
get_pool(void) {

  ufixnum s;

#ifndef NO_FILE_LOCKING
  if (multiprocess_memory_pool) {

    open_pool();
    lock_pool();
    s=Pool->s;
    unlock_pool();
    
  } else
#endif
    
    s=data_pages();

  return s;
  
}

static void
pool_stat(void) {

#ifndef NO_FILE_LOCKING
  if (multiprocess_memory_pool) {

    open_pool();
    lock_pool();
    pool_pid=Pool->pid;
    pool_n=Pool->n;
    pool_s=Pool->s;
    unlock_pool();

  }
#endif

}


static void
pool_check(void) {

  /* if (pool!=-1) */
  /*   massert(get_pool()==data_pages() */
  /* 	    ||!fprintf(stderr,"%lu %lu %lu\n",get_pool(),page((void *)heap_end-data_start),page(((rb_end-rb_start))))); */

}
