def export_csv(table):
    """UTF-8 flat monitoring tables, with stable column order."""
    return table.to_csv(index=False).encode('utf-8')
