/*! \file  PreAMGUtil.inl
 *
 *  \brief Utilities for link list data structure
 *
 *  \note  This file contains Level-4 (Pre) functions, which are used in:
 *         PreAMGCoarsenRS.c
 *
 *  Adapted from hypre 2.0 by Xuehai Huang, 09/06/2009
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#define LIST_HEAD -1 /**< head of the linked list */
#define LIST_TAIL -2 /**< tail of the linked list */

/**
 * \struct Link
 * \brief Struct for Links
 */
typedef struct
{
    
    //! previous node in the linklist
    INT prev;
    
    //! next node in the linklist
    INT next;
    
} Link; /**< General data structure for Links */

/**
 * \struct linked_list
 * \brief A linked list node
 *
 * \note This definition is adapted from hypre 2.0.
 */
typedef struct linked_list
{
    
    //! data
    INT data;
    
    //! starting of the list
    INT head;
    
    //! ending of the list
    INT tail;
    
    //! next node
    struct linked_list *next_node;
    
    //! previous node
    struct linked_list *prev_node;
    
} ListElement; /**< Linked element in list */

/**
 * List of links
 */
typedef ListElement *LinkList; /**< linked list */

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

#ifndef AMG_COARSEN_CR /* the following code is not needed in CR AMG */

/**
 * \fn static void dispose_node( LinkList node_ptr )
 *
 * \brief Free memory space used by node_ptr
 *
 * \param node_ptr   Pointer to the node in linked list
 *
 * \author Xuehai Huang
 * \date   09/06/2009
 */
static void dispose_node (LinkList node_ptr)
{
    fasp_mem_free(node_ptr); node_ptr = NULL;
}

/**
 * \fn static LinkList create_node (INT Item)
 *
 * \brief Create an node using Item for its data field
 *
 * \author Xuehai Huang
 * \date   09/06/2009
 */
static LinkList create_node (INT Item)
{
    LinkList new_node_ptr;
    
    /* Allocate memory space for the new node.
     * return with error if no space available
     */
    new_node_ptr = (LinkList) fasp_mem_calloc(1,sizeof(ListElement));
    
    new_node_ptr -> data = Item;
    new_node_ptr -> next_node = NULL;
    new_node_ptr -> prev_node = NULL;
    new_node_ptr -> head = LIST_TAIL;
    new_node_ptr -> tail = LIST_HEAD;
    
    return (new_node_ptr);
}

/**
 * \fn static void remove_node (LinkList *LoL_head_ptr, LinkList *LoL_tail_ptr,
 *                              INT measure, INT index, INT *lists, INT *where)
 *
 * \brief Removes a point from the lists
 *
 * \author Xuehai Huang
 * \date   09/06/2009
 */
static void remove_node (LinkList *LoL_head_ptr,
                         LinkList *LoL_tail_ptr,
                         INT measure,
                         INT index,
                         INT *lists,
                         INT *where)
{
    LinkList LoL_head = *LoL_head_ptr;
    LinkList LoL_tail = *LoL_tail_ptr;
    LinkList list_ptr = LoL_head;

    do {
        if (measure == list_ptr->data) {
            /* point to be removed is only point on list,
             which must be destroyed */
            if (list_ptr->head == index && list_ptr->tail == index) {
                /* removing only list, so num_left better be 0! */
                if (list_ptr == LoL_head && list_ptr == LoL_tail) {
                    LoL_head = NULL;
                    LoL_tail = NULL;
                    dispose_node(list_ptr);

                    *LoL_head_ptr = LoL_head;
                    *LoL_tail_ptr = LoL_tail;
                    return;
                }
                else if (LoL_head == list_ptr) { /*removing 1st (max_measure) list */
                    list_ptr -> next_node -> prev_node = NULL;
                    LoL_head = list_ptr->next_node;
                    dispose_node(list_ptr);

                    *LoL_head_ptr = LoL_head;
                    *LoL_tail_ptr = LoL_tail;
                    return;
                }
                else if (LoL_tail == list_ptr) { /* removing last list */
                    list_ptr -> prev_node -> next_node = NULL;
                    LoL_tail = list_ptr->prev_node;
                    dispose_node(list_ptr);

                    *LoL_head_ptr = LoL_head;
                    *LoL_tail_ptr = LoL_tail;
                    return;
                }
                else {
                    list_ptr -> next_node -> prev_node = list_ptr -> prev_node;
                    list_ptr -> prev_node -> next_node = list_ptr -> next_node;
                    dispose_node(list_ptr);

                    *LoL_head_ptr = LoL_head;
                    *LoL_tail_ptr = LoL_tail;
                    return;
                }
            }
            else if (list_ptr->head == index) { /* index is head of list */
                list_ptr->head = lists[index];
                where[lists[index]] = LIST_HEAD;
                return;
            }
            else if (list_ptr->tail == index) { /* index is tail of list */
                list_ptr->tail = where[index];
                lists[where[index]] = LIST_TAIL;
                return;
            }
            else { /* index is in middle of list */
                lists[where[index]] = lists[index];
                where[lists[index]] = where[index];
                return;
            }
        }
        list_ptr = list_ptr -> next_node;
    } while (list_ptr != NULL);

    printf("### ERROR: This list is empty! [%s:%d]\n", __FILE__, __LINE__);
    return;
}

/**
 * \fn static void enter_list (LinkList *LoL_head_ptr, LinkList *LoL_tail_ptr,
 *                             INT measure, INT index, INT *lists, INT *where)
 *
 * \brief Places point in new list
 *
 * \author Xuehai Huang
 * \date   09/06/2009
 */
static void enter_list (LinkList *LoL_head_ptr,
                        LinkList *LoL_tail_ptr,
                        INT measure,
                        INT index,
                        INT *lists,
                        INT *where)
{
    LinkList   LoL_head = *LoL_head_ptr;
    LinkList   LoL_tail = *LoL_tail_ptr;
    LinkList   list_ptr;
    LinkList   new_ptr;
    
    INT        old_tail;
    
    list_ptr =  LoL_head;
    
    if (LoL_head == NULL) { /* no lists exist yet */
        new_ptr = create_node(measure);
        new_ptr->head = index;
        new_ptr->tail = index;
        lists[index] = LIST_TAIL;
        where[index] = LIST_HEAD;
        LoL_head = new_ptr;
        LoL_tail = new_ptr;
        
        *LoL_head_ptr = LoL_head;
        *LoL_tail_ptr = LoL_tail;
        return;
    }
    else {
        do {
            if (measure > list_ptr->data) {
                new_ptr = create_node(measure);
                
                new_ptr->head = index;
                new_ptr->tail = index;
                
                lists[index] = LIST_TAIL;
                where[index] = LIST_HEAD;
                
                if ( list_ptr->prev_node != NULL) {
                    new_ptr->prev_node             = list_ptr->prev_node;
                    list_ptr->prev_node->next_node = new_ptr;
                    list_ptr->prev_node            = new_ptr;
                    new_ptr->next_node             = list_ptr;
                }
                else {
                    new_ptr->next_node  = list_ptr;
                    list_ptr->prev_node = new_ptr;
                    new_ptr->prev_node  = NULL;
                    LoL_head            = new_ptr;
                }
                
                *LoL_head_ptr = LoL_head;
                *LoL_tail_ptr = LoL_tail;
                return;
            }
            else if (measure == list_ptr->data) {
                old_tail        = list_ptr->tail;
                lists[old_tail] = index;
                where[index]    = old_tail;
                lists[index]    = LIST_TAIL;
                list_ptr->tail  = index;
                return;
            }
            
            list_ptr = list_ptr->next_node;
        } while (list_ptr != NULL);
        
        new_ptr = create_node(measure);
        new_ptr->head = index;
        new_ptr->tail = index;
        lists[index] = LIST_TAIL;
        where[index] = LIST_HEAD;
        LoL_tail->next_node = new_ptr;
        new_ptr->prev_node = LoL_tail;
        new_ptr->next_node = NULL;
        LoL_tail = new_ptr;
        
        *LoL_head_ptr = LoL_head;
        *LoL_tail_ptr = LoL_tail;
        return;
    }
}

#endif

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
