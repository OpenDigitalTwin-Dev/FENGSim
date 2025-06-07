
#define MAGIC1 ''
#define MAGIC2 'A'


/*                      SIZE in BYTES 10+N
   magic1                1
   magic2                1
   type (id)             1    the TYPE of message.  callback, command, etc...[an enum!]
   flag                  1    things like, do acknowledge, etc. 
   size of actual_body   3    N Use PUSH_LONG to store, POP_LONG to read 
   msg_index             3    counter inc'd on each message sent, PUSH_SHORT to write.. 
   actual_body           N    data 
*/

enum mtype {
  m_not_used,
  m_create_command,
  m_reply,
  m_call,
  m_tcl_command,
  m_tcl_command_wait_response,
  m_tcl_clear_connection,        /* clear tk connection and command buff */
  m_tcl_link_text_variable,
  m_set_lisp_loc,
  m_tcl_set_text_variable,
  m_tcl_unlink_text_variable
};
  
struct message_header {
  char magic1;
  char magic2;
  char type;
  unsigned char flag;
  unsigned char size[3];
  unsigned char msg_id[3];
  char body[1];
};
  
#ifndef SIGNAL_PARENT_WAITING_RESPONSE
#define SIGNAL_PARENT_WAITING_RESPONSE 1
#endif




#define BYTE_S 8
#define BYTE_MASK (~(~0UL << BYTE_S))

#define GET_3BYTES(p,ans) do{ unsigned char* __p = (unsigned char *) p; \
				ans = BYTE_MASK&(*__p++); \
			  ans += (BYTE_MASK&((*__p++)))<<1*BYTE_S; \
			  ans += (BYTE_MASK&((*__p++)))<<2*BYTE_S;} while(0)

#define GET_2BYTES(p,ans) do{ unsigned char* __p = (unsigned char *) p; \
				ans = BYTE_MASK&(*__p++); \
			  ans += (BYTE_MASK&((*__p++)))<<1*BYTE_S; \
			  } while(0)


/* store an unsigned int n into the character pointer so that
   low order byte occurs first */

#define STORE_2BYTES(p,n)  do{ unsigned char* __p = (unsigned char *) p; \
				 *__p++ =  (n & BYTE_MASK);\
				 *__p++ = ((n >> BYTE_S) & BYTE_MASK); \
				 }\
                                  while (0)

#define STORE_3BYTES(p,n)  do{ unsigned char* __p = (unsigned char *) p; \
				 *__p++ =  (n & BYTE_MASK);\
				 *__p++ = ((n >> BYTE_S) & BYTE_MASK); \
				 *__p++ = ((n >> (2*BYTE_S)) & BYTE_MASK);}\
                                  while (0)
#define MESSAGE_HEADER_SIZE 10


#define HDR_SIZE 5
struct our_header
{ unsigned char magic;
  unsigned char length[2];  /* length of packet including HDR_SIZE */
  unsigned char received[2];  /* tell other side about how many bytes received.
		      incrementally */
};

struct connection_state
{ int fd;
  int total_bytes_sent;
  int total_bytes_received;
  int bytes_sent_not_received;
  int bytes_received_not_confirmed;
  int next_packet_offset;  /* offset from valid_data for start of next packet*/
  char *read_buffer;
  int read_buffer_size;
  char *valid_data;
  int  valid_data_size;
  int max_allowed_in_pipe;
  int write_timeout;
};

#define MAX_ALLOWED_IN_PIPE PAGESIZE
#define READ_BUFF_SIZE (PAGESIZE<<1)

extern struct connection_state *dsfd;

#define fScheck_dsfd_for_input(sf,timeout) \
  (sf->valid_data_size > 0 ? 1 : (fixnum)fScheck_fd_for_input(sf->fd,timeout))

#define OBJ_TO_CONNECTION_STATE(x) \
  ((struct connection_state *)(void *)((x)->ust.ust_self))

struct connection_state * setup_connection_state();
