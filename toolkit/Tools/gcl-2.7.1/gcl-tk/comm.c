
#include <errno.h>


#ifndef NO_DEFUN
#ifndef DEFUN
#define DEFUN(string,ret,fname,pack,min,max, flags, ret0a0,a12,a34,a56,doc) ret fname
#endif
#endif


#ifndef HZ
#define HZ 60
#endif

#ifndef SET_TIMEVAL
#define SET_TIMEVAL(t,timeout) \
  t.tv_sec = timeout/HZ; t.tv_usec = (int) ((timeout%HZ)*(1000000.0)/HZ)
#endif


DEFUN("CHECK-FD-FOR-INPUT",object,fScheck_fd_for_input,SI,2,2,NONE,II,IO,OO,OO,(fixnum fd,fixnum timeout),
      "Check FD a file descriptor for data to read, waiting TIMEOUT clicks \
for data to become available.  Here there are \
INTERNAL-TIME-UNITS-PER-SECOND in one second.  Return is 1 if data \
available on FD, 0 if timeout reached and -1 if failed.")
{
  fd_set inp;
  int n;
  struct timeval t;

  SET_TIMEVAL(t,timeout);
  FD_ZERO(&inp);
  FD_SET(fd, &inp);
  n = select(fd + 1, &inp, NULL, NULL, &t);
  if (n < 0)
    return (object)-1;
  else if (FD_ISSET(fd, &inp))
    return (object)1;
  else
    return (object)0;
}
#ifdef STATIC_FUNCTION_POINTERS
object
fScheck_fd_for_input(fixnum fd,fixnum timeout) {
  return FFN(fScheck_fd_for_input)(fd,timeout);
}
#endif



#define MAX_PACKET 1000
#define MUST_CONFIRM 2000
#define OUR_SOCK_MAGIC 0206


/* Each write and read will be of a packet including information about
   how many we have read and written.
   Sometimes we must read more messages, in order to check whether
   the one being sent has info about bytes_received.
   */




struct connection_state *
setup_connection_state(int fd)
{ struct connection_state * res;
  res = (void *)malloc(sizeof(struct connection_state));
  bzero(res,sizeof(struct connection_state));
  res->fd = fd;
  res->read_buffer_size = READ_BUFF_SIZE;
  res->read_buffer = (void *)malloc(READ_BUFF_SIZE);
  res->valid_data = res->read_buffer;
  res->max_allowed_in_pipe = MAX_ALLOWED_IN_PIPE;
  res->write_timeout = 30* 100;
  return res;
}

/* P is supposed to start with a hdr  and run N bytes. */
static void
scan_headers(sfd)
     struct connection_state *sfd;
{ struct our_header *hdr;
  char *p = sfd->valid_data + sfd->next_packet_offset;
  int n = sfd->valid_data_size - sfd->next_packet_offset;
  int length,received;
  while (n >= HDR_SIZE)
    { hdr = (void *)p;
      if (hdr->magic != OUR_SOCK_MAGIC)
	abort();
      GET_2BYTES(&hdr->received, received);
      STORE_2BYTES(&hdr->received, 0);
      sfd->bytes_sent_not_received -= received;
      GET_2BYTES(&hdr->length, length);
      p += length;
      n -= length;
    }
}

static int
write1(struct connection_state *,const char *,int);


static void
send_confirmation(struct connection_state *sfd)
{ write1(sfd,0,0);
}


/* read from SFD to buffer P  M bytes.   Allow TIMEOUT
   delay while waiting for data to arrive.
   return number of bytes actually read.
   The data arrives on the pipe packetized, but is unpacketized
   by this function.    It gets info about bytes that have
   been received by the other process, and updates info in the state.

*/   

static int
read1(sfd,p,m,timeout)
struct connection_state* sfd;     
char *p;
int timeout;
int m;
{ int nread=0;
  int wanted = m;
  int length;
  struct our_header *hdr;
  if (wanted == 0)
    goto READ_SOME;
 TRY_PACKET:
  if (sfd->next_packet_offset > 0)
    { int mm = (sfd->next_packet_offset >= wanted ? wanted :
		sfd->next_packet_offset);
	{ bcopy(sfd->valid_data,p,mm);
	  p += mm;
	  sfd->valid_data+= mm;
	  sfd->valid_data_size -= mm;
	  sfd->next_packet_offset -= mm;

	}
      wanted -= mm;
      if (0 == wanted) return m;

    }
 /* at beginning of a packet */
	  
 if (sfd->valid_data_size >= HDR_SIZE)
   { hdr =  (void *) sfd->valid_data;
    GET_2BYTES(&hdr->length,length);
   }
  else goto READ_SOME;
  if (length > sfd->valid_data_size)
    goto READ_SOME;
  /* we have a full packet available */
  {int mm = (wanted <= length - HDR_SIZE ? wanted : length - HDR_SIZE);
   /* mm = amount to copy */
   	{ bcopy(sfd->valid_data+HDR_SIZE,p,mm);
	  p += mm;
	  sfd->valid_data+= (mm +HDR_SIZE);
	  sfd->valid_data_size -= (mm +HDR_SIZE);
	  sfd->next_packet_offset = length - (mm + HDR_SIZE);
	  wanted -= mm;
	}
    if (0 == wanted) return m;
   goto TRY_PACKET;
 }

 READ_SOME:
  if (sfd->read_buffer_size - sfd->valid_data_size < MAX_PACKET)
    { char *tmp ;
      tmp = (void *) malloc(2* sfd->read_buffer_size);
      if (tmp == 0) error("out of free space");
      bcopy(sfd->valid_data,tmp,sfd->valid_data_size);
      free(sfd->read_buffer);
      sfd->valid_data = sfd->read_buffer = tmp;
      sfd->read_buffer_size *= 2;
    }
  if(sfd->read_buffer_size - (sfd->valid_data - sfd->read_buffer) < MAX_PACKET)
    { bcopy(sfd->valid_data,sfd->read_buffer,sfd->valid_data_size);
      sfd->valid_data=sfd->read_buffer;}
   /* there is at least a packet size of space available */   
  if (((fixnum)(FFN(fScheck_fd_for_input)(sfd->fd,sfd->write_timeout))>0)) {
  again:
    {
      char *start = sfd->valid_data+sfd->valid_data_size;
      nread = SAFE_READ(sfd->fd,start,sfd->read_buffer_size - (start -  sfd->read_buffer));
      if (nread<0) {
	if (errno == EAGAIN) goto again;
	return -1;
      }
      if (nread == 0)  { 
	return 0;
      }
      sfd->total_bytes_received +=  nread;
      sfd->bytes_received_not_confirmed +=  nread;
      sfd->valid_data_size += nread; 
      if(sfd->bytes_received_not_confirmed > MUST_CONFIRM)
	send_confirmation(sfd);
      scan_headers(sfd); 
      goto TRY_PACKET;
    }
  }

  return 0;

}

/* send BYTES chars from buffer P to CONNECTION.
   They are packaged up with a hdr */

static void
write_timeout_error(char *);

static void
connection_failure(char *);

int
write1(sfd,p,bytes)
     struct connection_state *sfd;
     const char *p;
     int bytes;
{ 
  int bs;
  int to_send = bytes;
 BEGIN:
  bs = sfd->bytes_sent_not_received;
  if (bs  > sfd->max_allowed_in_pipe)
    {read1(sfd,0,0,sfd->write_timeout);
     if (bs > sfd->bytes_sent_not_received)
       goto BEGIN;
      write_timeout_error("");
    }
  {struct our_header *hdr;
   char buf[MAX_PACKET];
   int n_to_send =
     (bytes > MAX_PACKET -HDR_SIZE ? MAX_PACKET : bytes+HDR_SIZE); 
   hdr = (void *) buf;
   STORE_2BYTES(&hdr->length, n_to_send);
   hdr->magic = OUR_SOCK_MAGIC;
   STORE_2BYTES(&hdr->received, sfd->bytes_received_not_confirmed);
   sfd->bytes_received_not_confirmed =0;
   sfd->bytes_sent_not_received += n_to_send;
   bcopy(p, buf+HDR_SIZE,n_to_send - HDR_SIZE);

 AGAIN:
   { int n = write(sfd->fd,buf,n_to_send);
     if (n == n_to_send);
     else   if (n < 0)
       {  if (errno == EAGAIN)
	    { goto AGAIN;
	    }
       else connection_failure("");
	  }
     else abort();
   }
   p += (n_to_send -HDR_SIZE);
   bytes -= (n_to_send -HDR_SIZE);
   if (bytes==0) return to_send;
   goto BEGIN;
 }
      
}	  

DEFUN("CLEAR-CONNECTION",object,fSclear_connection,SI,1,1,NONE,II,OO,OO,OO,(fixnum fd),
      "Read on FD until nothing left to read.  Return number of bytes read") {
  
  char buffer[0x1000];
  int n=0;
  while ((fixnum)(FFN(fScheck_fd_for_input)(fd,0)))
    n+=read(fd,buffer,sizeof(buffer));
  
  return (object)(fixnum)n;

}
#ifdef STATIC_FUNCTION_POINTERS
object
fSclear_connection(fixnum fd) {
  return FFN(fSclear_connection)(fd);
}
#endif



