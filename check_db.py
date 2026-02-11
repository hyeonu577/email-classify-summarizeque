import os
import sqlite3
import hashlib
from datetime import datetime, timedelta

def get_db_path():
    """데이터베이스 경로 반환"""
    return os.path.join(os.path.dirname(__file__), "checked_items.db")

def print_items():
    """
    저장된 아이템 중 가장 최근 20개를 출력합니다.
    """
    db_path = get_db_path()
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT hash_value, title, created_at 
            FROM checked_items 
            ORDER BY created_at DESC 
            LIMIT 50
        ''')
        rows = cursor.fetchall()
    
    print(f"\n{'='*20} 최신 저장 목록 {'='*20}")
    if not rows:
        print("저장된 데이터가 없습니다.")
    
    for row in rows:
        h_val, title, c_at = row
        print(f"Hash : {h_val}")
        print(f"Title: {title}")
        print(f"Time : {c_at}")
        print("-" * 40)


def delete_item_by_hash(target_hash):
    """
    hash_value를 입력받아 해당 데이터를 삭제합니다.
    """
    db_path = get_db_path()
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM checked_items WHERE hash_value = ?', (target_hash,))
        conn.commit()
        
        if cursor.rowcount > 0:
            print(f"\n[성공] 해시값 '{target_hash}' 항목이 삭제되었습니다.")
        else:
            print(f"\n[실패] 해시값 '{target_hash}'를 찾을 수 없습니다.")


def delete_recent_items(days=2):
    """
    현재 시간으로부터 지정된 일수(days) 이내에 생성된 데이터를 삭제합니다.
    """
    db_path = get_db_path()
    
    # 1. 기준 시간 계산 (현재 시간 - 2일)
    cutoff_date = datetime.now() - timedelta(days=days)
    # DB에 저장된 형식과 맞추기 위해 문자열로 변환 (예: '2023-10-27 10:00:00')
    cutoff_str = cutoff_date.strftime('%Y-%m-%dT%H:%M:%S.%f')
    
    print(f"\n[작업 시작] {cutoff_str} 이후에 저장된 데이터를 삭제합니다...")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # 2. 기준 시간보다 크거나 같은(>=) created_at을 가진 행 삭제
        cursor.execute('DELETE FROM checked_items WHERE created_at >= ?', (cutoff_str,))
        conn.commit()
        
        if cursor.rowcount > 0:
            print(f"[성공] 총 {cursor.rowcount}개의 항목이 삭제되었습니다.")
        else:
            print(f"[알림] 해당 기간 내에 삭제할 데이터가 없습니다.")


if __name__ == "__main__":
    # for to_be_deleted_hash in ['caccbd0b8889b3066026015b2f59d189']:
    #     delete_item_by_hash(to_be_deleted_hash)
    print_items()
    delete_recent_items(days=2)
    print_items()
    